from torch import nn


class Audio2Feature(nn.Module):
    '''
    audio to face (only-mouth) delta learning 
    '''
    def __init__(self, cfg):
        super(Audio2Feature, self).__init__()
        
        self.cfg = cfg
        
        self.downsample = nn.Sequential(
            nn.Linear(in_features=cfg.APC_hidden_size * 2, out_features=cfg.APC_hidden_size),
            nn.BatchNorm1d(cfg.APC_hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.APC_hidden_size, cfg.APC_hidden_size)
        )
        
        self.LSTM = nn.LSTM(input_size=cfg.APC_hidden_size,
                            hidden_size=256,
                            num_layers=3,
                            dropout=0,
                            bidirectional=False,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, cfg.output_size)
        )
        
    
    def forward(self, audio_features):
        '''
        audio_features: [b, T, ndim] check TODO 
        '''
        bs, item_len, ndim = audio_features.shape
        audio_features = audio_features.reshape(bs, -1, ndim*2)
        down_audio_feats = self.downsample(audio_features.reshape(-1, ndim*2)).reshape(bs, int(item_len/2), ndim)
        output, (hn, cn) = self.LSTM(down_audio_feats)
        pred = self.fc(output.reshape(-1, 256)).reshape(bs, int(item_len/2), -1)
        
        return pred
