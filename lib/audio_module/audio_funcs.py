import math
import torch
import librosa
from librosa.filters import mel
from torch import nn
from torch.nn import functional as F 


class Audio2Mel(nn.Module):
    def __init__(self, 
                 n_fft=512,
                 hop_length=256,
                 win_length=1024,
                 sampling_rate=16000,
                 n_mel_channels=80,
                 mel_fmin=90,
                 mel_fmax=7600.0):
        super(Audio2Mel, self).__init__()
        
        window = torch.hann_window(win_length).float()
        mel_basis = mel(sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax)
        
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.min_mel = math.log(1e-5)
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        
    
    def forward(self, audio, normalize=True):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, center=False)
        real_part, img_part = fft.unbind(-1)
        mangnitute = torch.sqrt(real_part ** 2 + img_part ** 2)
        mel_output = torch.matmul(self.mel_basis, mangnitute)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=1e-5))
        
        # print(log_mel_spec, 'log mel', mel_output, 'not log')
        
        min_log_mel = torch.min(log_mel_spec)
        max_log_mel = torch.min(log_mel_spec)
        
        log_mel_mean = torch.mean(log_mel_spec)
        log_mel_std = torch.std(log_mel_spec)
        
        if normalize:
            log_mel_spec = (log_mel_spec - self.min_mel) / -self.min_mel  # original normal, right
            # log_mel_spec = torch.subtract(log_mel_spec, torch.mean(log_mel_spec)) / (max_log_mel - min_log_mel + 1e-7)  # mean normal
            # log_mel_spec = (log_mel_spec - min_log_mel) / (max_log_mel - min_log_mel + 1e-7)  # min-max normal
            # log_mel_spec = (log_mel_spec - log_mel_mean) / (torch.sqrt(log_mel_std))  # 0-1 gaussion
            # print('normalized', log_mel_spec)
            return log_mel_spec
        
    def mel_to_audio(self, mel):
        mel = torch.exp(mel * (-self.min_mel) + self.min_mel) ** 2
        mel_np = mel.cpu().numpy()
        audio = librosa.feature.inverse.mel_to_audio(mel_np, sr=self.sampling_rate, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                                                     window='hann', center=False, pad_mode='reflect', power=2.0, n_iter=32, fmin=self.mel_fmin, fmax=self.mel_fmax)
        
        return audio
