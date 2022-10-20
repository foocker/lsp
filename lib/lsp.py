from torch import nn
from lib.models.audio2feature import Audio2Feature
from lib.models.audio2headpose import Audio2Headpose
from lib.models.feature2face import Feature2Face
from lib.models.model_utils import copy_state_dict

from lib.solover.losses import Sample_GMM
from tqdm import tqdm

from torch.cuda.amp import autocast
import numpy as np
import torch

import os

class LSP(nn.Module):
    def __init__(self, cfg=None):
        super(LSP, self).__init__()
        self.cfg = cfg
        self.a2f = Audio2Feature(cfg)
        self.a2h = Audio2Headpose(cfg)
        self.f2f = Feature2Face(cfg)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        if os.path.exists(self.cfg.infer_file_g) and self.cfg.infer_g:
            self.load_g_weight()
        if os.path.exists(self.cfg.infer_file_h) and self.cfg.infer_h:
            self.load_h_weight()
    
    def audio2mouth(self):
        pass
    
    def audio2headpose(self):
        pass
    
    def landmarks2face(self):
        pass
    
    def model_dict(self):
        return {
            'audio2mouth': self.a2f.state_dict(),
            'audio2headpose': self.a2h.state_dict(),
            'feature2face_d': self.f2f.FFD.state_dict(),
            'feature2face_g': self.f2f.FFG.state_dict()
        }
        
    def load_g_weight(self):
        weights = torch.load(self.cfg.infer_file_g)
        weights_f2f_g = weights['feature2face_g']
        
        copy_state_dict(self.f2f.FFG.state_dict(), weights_f2f_g)

        self.f2f.to(self.device)
        
        self.f2f.eval()
        
    def load_h_weight(self):
        weights = torch.load(self.cfg.infer_file_h)
        weights_a2h = weights['audio2headpose']
        
        copy_state_dict(self.a2h.state_dict(), weights_a2h)

        self.a2h.to(self.device)
        
        self.a2h.eval()
    
    def inference(self, audio_feats, *args):
        
        weights = torch.load(self.cfg.infer_file_a)
        weights_a2f = weights['audio2mouth']
        # for k, v in weights_a2f.items():
        #     if k == 'fc.4.weight':
        #         print(v)
        #         break
        
        copy_state_dict(self.a2f.state_dict(), weights_a2f)
        # for name, w in self.a2f.named_parameters():
        #     print(name)
        #     if name == 'fc.4.weight':
        #         print(w)
        #         break
        self.a2f.to(self.device)
        self.a2f.eval()
        
        if len(args) == 0:
            frame_future = self.cfg.frame_future
            nframe = int(audio_feats.shape[0] / 2) 
            
            if not frame_future == 0:
                audio_feats_insert = np.repeat(audio_feats[-1],  2 * (frame_future)).reshape(-1, 2 * (frame_future)).T
                audio_feats = np.concatenate([audio_feats, audio_feats_insert])

            with torch.no_grad():
                input = torch.from_numpy(audio_feats).unsqueeze(0).float().to(self.device)
                preds = self.a2f(input)
                
            # drop first frame future results
            if not frame_future == 0:
                preds = preds[0, frame_future:].cpu().detach().numpy()
            else:
                preds = preds[0, :].cpu().detach().numpy()
            
            assert preds.shape[0] == nframe
                                
            return preds
        else:
            pass
        
    def inference_h(self, audio_feats, pre_headpose, fill_zero=True, sigma_scale=0.0, cfg=None):
        ''' generate landmark sequences given audio and a initialized landmark.
        Note that the audio input should have the same sample rate as the training.
        Args:
            audio_sequences: [n,], in numpy
            init_landmarks: [npts, 2], in numpy
            sample_rate: audio sample rate, should be same as training process.
            method(str): optional, how to generate the sequence, indeed it is the 
                loss function during training process. Options are 'L2' or 'GMM'.
        Reutrns:
            landmark_sequences: [T, npts, 2] predition landmark sequences
        '''

        frame_future = cfg.frame_future
        audio_feats = audio_feats.reshape(-1, 512 * 2)
        nframe = audio_feats.shape[0] - frame_future
        pred_headpose = np.zeros([nframe, cfg.A2H_GMM_ndim])

        # fill zero or not
        if fill_zero == True:
            # headpose
            audio_feats_insert = np.repeat(audio_feats[0], cfg.A2H_receptive_field - 1)
            audio_feats_insert = audio_feats_insert.reshape(-1, cfg.A2H_receptive_field - 1).T
            audio_feats = np.concatenate([audio_feats_insert, audio_feats])
            # history headpose
            history_headpose = np.repeat(pre_headpose, cfg.A2H_receptive_field)
            history_headpose = history_headpose.reshape(-1, cfg.A2H_receptive_field).T
            history_headpose = torch.from_numpy(history_headpose).unsqueeze(0).float().to(self.device)
            infer_start = 0   
        else:
            return None

        with torch.no_grad():
            for i in tqdm(range(infer_start, nframe), desc='generating headpose'):
                history_start = i - infer_start
                input_audio_feats = audio_feats[history_start + frame_future: history_start + frame_future + cfg.A2H_receptive_field]
                input_audio_feats = torch.from_numpy(input_audio_feats).unsqueeze(0).float().to(self.device)
                
                preds = self.a2h.forward(history_headpose, input_audio_feats)
                
                
                pred_data = Sample_GMM(preds, cfg.A2H_GMM_ncenter, cfg.A2H_GMM_ndim, sigma_scale=sigma_scale)
                # pred_data = preds
                    
                # get predictions
                pred_headpose[i] = pred_data[0,0].cpu().detach().numpy()  
                history_headpose = torch.cat((history_headpose[:,1:,:], pred_data.to(self.device)), dim=1)  # add in time-axis                
                
        return pred_headpose
    
    
    def inference_g(self, feature_map, cand_image):
        
        with torch.no_grad():      
            if cand_image == None:
                input_feature_maps = feature_map
            else:
                input_feature_maps = torch.cat([feature_map, cand_image], dim=1)
            if not self.cfg.fp16:
                fake_pred = self.f2f.FFG(input_feature_maps)          
            else:
                with autocast():
                    fake_pred = self.f2f.FFG(input_feature_maps)
        return fake_pred