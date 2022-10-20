from torch import nn
from .networks import WaveNet


class Audio2Headpose(nn.Module):
    def __init__(self, cfg):
        super(Audio2Headpose, self).__init__()
        self.cfg = cfg
        output_size = (2 * cfg.A2H_GMM_ndim + 1) * cfg.A2H_GMM_ncenter
        
        self.audio_downsample = nn.Sequential(
            nn.Linear(in_features=cfg.APC_hidden_size * 2, out_features=cfg.APC_hidden_size),
            nn.BatchNorm1d(cfg.APC_hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.APC_hidden_size, cfg.APC_hidden_size)
        )
        self.WaveNet = WaveNet(cfg.A2H_wavenet_residual_layers,
                               cfg.A2H_wavenet_residual_blocks,
                               cfg.A2H_wavenet_residual_channels,
                               cfg.A2H_wavenet_dilation_channels,
                               cfg.A2H_wavenet_skip_channels,  # 256
                               cfg.A2H_wavenet_kernel_size,
                               cfg.time_frame_length,  # 240
                               cfg.A2H_wavenet_use_bias,
                               True,
                               cfg.A2H_wavenet_input_channels,  # 12
                               cfg.A2H_GMM_ncenter,  # 1
                               cfg.A2H_GMM_ndim,  # 12
                               output_size,
                               cfg.A2H_wavenet_cond_channels)  # 512
        
        self.item_length = self.WaveNet.receptive_field + cfg.time_frame_length - 1
        
    
    def forward(self, history_info, audio_features):
        bs, item_len, ndim = audio_features.shape
        down_audio_feats = self.audio_downsample(audio_features.reshape(-1, ndim)).reshape(bs, item_len, -1)
        pred = self.WaveNet.forward(history_info.permute(0,2,1), down_audio_feats.transpose(1,2)) 
        return pred