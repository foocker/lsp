import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class APC_encoder(nn.Module):
    def __init__(self, mel_dim, hidden_size, num_layers, residual):
        super(APC_encoder, self).__init__()
        
        input_size = mel_dim
        in_sizes = ([input_size] + [hidden_size] * (num_layers - 1))
        out_sizes = [hidden_size] * num_layers
        
        self.rnns = nn.ModuleList([nn.GRU(input_size=in_size, hidden_size=out_size, batch_first=True) for (in_size, out_size) in zip(in_sizes, out_sizes)])
        
        self.rnn_residual = residual
        
    def forward(self, inputs, lengths):
        '''
        input:
            inputs: (batch_size, seq_len, mel_dim)
            lengths: (batch_size,)

        return:
            predicted_mel: (batch_size, seq_len, mel_dim)
            internal_reps: (num_layers + x, batch_size, seq_len, rnn_hidden_size),
            where x is 1 if there's a prenet, otherwise 0
        '''
        with torch.no_grad():
            seq_len = inputs.size(1)
            packed_rnn_inputs = pack_padded_sequence(inputs, lengths, True)
        
            for i, layer in enumerate(self.rnns):
                packed_rnn_outputs, _ = layer(packed_rnn_inputs)
                
                rnn_outputs, _ = pad_packed_sequence(
                        packed_rnn_outputs, True, total_length=seq_len)
                # outputs: (batch_size, seq_len, rnn_hidden_size)
                
                if i + 1 < len(self.rnns):
                    rnn_inputs, _ = pad_packed_sequence(
                            packed_rnn_inputs, True, total_length=seq_len)
                    # rnn_inputs: (batch_size, seq_len, rnn_hidden_size)
                    if self.rnn_residual and rnn_inputs.size(-1) == rnn_outputs.size(-1):
                        # Residual connections
                        rnn_outputs = rnn_outputs + rnn_inputs
                    packed_rnn_inputs = pack_padded_sequence(rnn_outputs, lengths, True)
        
        
        return rnn_outputs


class WaveNet(nn.Module):
    ''' This is a complete implementation of WaveNet architecture, mainly composed
    of several residual blocks and some other operations.
    Args:
        batch_size: number of batch size
        residual_layers: number of layers in each residual blocks
        residual_blocks: number of residual blocks
        dilation_channels: number of channels for the dilated convolution
        residual_channels: number of channels for the residual connections
        skip_channels: number of channels for the skip connections
        end_channels: number of channels for the end convolution
        classes: Number of possible values each sample can have as output
        kernel_size: size of dilation convolution kernel
        output_length(int): Number of samples that are generated for each input
        use_bias: whether bias is used in each layer.
        cond(bool): whether condition information are applied. if cond == True:
            cond_channels: channel number of condition information
        `` loss(str): GMM loss is adopted. ``
    '''
    def __init__(self,
                 residual_layers = 10,
                 residual_blocks = 3,
                 dilation_channels = 32,
                 residual_channels = 32,
                 skip_channels = 256,
                 kernel_size = 2,
                 output_length = 16,
                 use_bias = False,
                 cond = True,
                 input_channels = 128,
                 ncenter = 1,
                 ndim = 68*2,
                 output_channels = 68*3,
                 cond_channels = 256,
                 activation = 'leakyrelu'):
        super(WaveNet, self).__init__()
        
        self.layers = residual_layers
        self.blocks = residual_blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.input_channels = input_channels
        self.ncenter = ncenter
        self.ndim = ndim
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.output_length = output_length
        self.bias = use_bias
        self.cond = cond
        self.cond_channels = cond_channels
        
        # build modules
        self.dilations = []
        self.dilation_queues = []
        residual_blocks = []
        self.receptive_field = 1
        
        # 1x1 convolution to create channels
        self.start_conv1 = nn.Conv1d(in_channels=self.input_channels,
                                     out_channels=self.residual_channels,
                                     kernel_size=1,
                                     bias=True)
        self.start_conv2 = nn.Conv1d(in_channels=self.residual_channels,
                                     out_channels=self.residual_channels,
                                     kernel_size=1,
                                     bias=True)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        self.drop_out2D = nn.Dropout2d(p=0.5)
        
        
        # build residual blocks
        for b in range(self.blocks):
            new_dilation = 1
            additional_scope = kernel_size - 1
            for i in range(self.layers):
                # create current residual block
                residual_blocks.append(residual_block(dilation = new_dilation,
                                                      dilation_channels = self.dilation_channels,
                                                      residual_channels = self.residual_channels,
                                                      skip_channels = self.skip_channels,
                                                      kernel_size = self.kernel_size,
                                                      use_bias = self.bias,
                                                      cond = self.cond,
                                                      cond_channels = self.cond_channels))
                new_dilation *= 2
                
                self.receptive_field += additional_scope
                additional_scope *= 2
        
        self.residual_blocks = nn.ModuleList(residual_blocks)
        # end convolutions
        
        self.end_conv_1 = nn.Conv1d(in_channels = self.skip_channels,
                                    out_channels = self.output_channels,
                                    kernel_size = 1,
                                    bias = True)
        self.end_conv_2 = nn.Conv1d(in_channels = self.output_channels,
                                    out_channels = self.output_channels,
                                    kernel_size = 1,
                                    bias = True)
        
    
    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s
    
    def forward(self, input, cond=None):
        '''
        Args:
            input: [b, ndim, T]
            cond: [b, nfeature, T]
        Returns:
            res: [b, T, ndim]
        '''
        # dropout
        x = self.drop_out2D(input)
        
        # preprocess
        x = self.activation(self.start_conv1(x))
        x = self.activation(self.start_conv2(x))
        skip = 0
#        for i in range(self.blocks * self.layers):
        for i, dilation_block in enumerate(self.residual_blocks):
            x, current_skip = self.residual_blocks[i](x, cond)
            skip += current_skip
        
        # postprocess
        res = self.end_conv_1(self.activation(skip))
        res = self.end_conv_2(self.activation(res))
        
        # cut the output size
        res = res[:, :, -self.output_length:]  # [b, ndim, T]
        res = res.transpose(1, 2)  # [b, T, ndim]
        
        return res
    

class residual_block(nn.Module):
    
    
    '''
    This is the implementation of a residual block in wavenet model. Every
    residual block takes previous block's output as input. The forward pass of 
    each residual block can be illusatrated as below:
        
    ######################### Current Residual Block ##########################
    #     |-----------------------*residual*--------------------|             #
    #     |                                                     |             # 
    #     |        |-- dilated conv -- tanh --|                 |             #
    # -> -|-- pad--|                          * ---- |-- 1x1 -- + --> *input* #
    #              |-- dilated conv -- sigm --|      |                        #
    #                                               1x1                       # 
    #                                                |                        # 
    # ---------------------------------------------> + -------------> *skip*  #
    ###########################################################################
    As shown above, each residual block returns two value: 'input' and 'skip':
        'input' is indeed this block's output and also is the next block's input.
        'skip' is the skip data which will be added finally to compute the prediction.
    The input args own the same meaning in the WaveNet class.
    
    '''
    def __init__(self,
                 dilation,
                 dilation_channels = 32,
                 residual_channels = 32,
                 skip_channels = 256,
                 kernel_size = 2,
                 use_bias = False,
                 cond = True,
                 cond_channels = 128):
        super(residual_block, self).__init__()
        
        self.dilation = dilation
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.bias = use_bias
        self.cond = cond
        self.cond_channels = cond_channels
        # zero padding to the left of the sequence.
        self.padding = (int((self.kernel_size - 1) * self.dilation), 0)
        
        # dilated convolutions
        self.filter_conv= nn.Conv1d(in_channels = self.residual_channels,
                                    out_channels = self.dilation_channels,
                                    kernel_size = self.kernel_size,
                                    dilation = self.dilation,
                                    bias = self.bias)
                
        self.gate_conv = nn.Conv1d(in_channels = self.residual_channels,
                                   out_channels = self.dilation_channels,
                                   kernel_size = self.kernel_size,
                                   dilation = self.dilation,
                                   bias = self.bias)
                
        # 1x1 convolution for residual connections
        self.residual_conv = nn.Conv1d(in_channels = self.dilation_channels,
                                       out_channels = self.residual_channels,
                                       kernel_size = 1,
                                       bias = self.bias)
                
        # 1x1 convolution for skip connections
        self.skip_conv = nn.Conv1d(in_channels = self.dilation_channels,
                                   out_channels = self.skip_channels,
                                   kernel_size = 1,
                                   bias = self.bias)
        
        # condition conv, no dilation
        if self.cond == True:
            self.cond_filter_conv = nn.Conv1d(in_channels = self.cond_channels,
                                    out_channels = self.dilation_channels,
                                    kernel_size = 1,
                                    bias = True)
            self.cond_gate_conv = nn.Conv1d(in_channels = self.cond_channels,
                                   out_channels = self.dilation_channels,
                                   kernel_size = 1,
                                   bias = True)
        
    
    def forward(self, input, cond=None):
        if self.cond is True and cond is None:
            raise RuntimeError("set using condition to true, but no cond tensor inputed")
            
        x_pad = F.pad(input, self.padding)
        # filter
        filter = self.filter_conv(x_pad)
        # gate
        gate = self.gate_conv(x_pad)
        
        if self.cond == True and cond is not None:
            filter_cond = self.cond_filter_conv(cond)
            gate_cond = self.cond_gate_conv(cond)
            # add cond results
            filter = filter + filter_cond
            gate = gate + gate_cond
                       
        # element-wise multiple
        filter = torch.tanh(filter)
        gate = torch.sigmoid(gate)
        x = filter * gate
        
        # residual and skip
        residual = self.residual_conv(x) + input
        skip = self.skip_conv(x)
               
        
        return residual, skip
    

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out