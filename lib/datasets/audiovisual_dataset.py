import torch

from .base_dataset import BaseDataset
import librosa
import os
import numpy as np

from lib.models.networks import APC_encoder
from lib.audio_module.utils import compute_mel_one_sequence


class AudioVisualDataset(BaseDataset):
    '''
    audio-visual dataset. currently, return 2D info and 3D tracking info.
    '''
    def __init__(self, cfg) -> None:
        BaseDataset.__init__(self, cfg)
        self.data3d_label_dir = os.path.join(cfg.data_dir, cfg.d3label_dir)
        self.audio_file = os.path.join(cfg.data_dir, cfg.original_dir, 'audio.wav')
        self.APC_dir = os.path.join(cfg.data_dir, cfg.APC_dir)
        self.APC_model_path = os.path.join(os.path.dirname(cfg.data_dir), 'APC_epoch_160.model')
        
        self.isTrain = self.cfg.isTrain
        
        self.sample_rate = cfg.sample_rate
        self.target_length = cfg.time_frame_length
        self.fps = cfg.fps
        
        self.audioRF_history = cfg.audioRF_history
        self.audioRF_future = cfg.audioRF_future
        self.compute_mel_online = cfg.compute_mel_online
        
        self.audio_samples_one_frame = self.sample_rate / self.fps
        self.frame_jump_stride = cfg.frame_jump_stride
        self.task = cfg.task
        self.item_length_audio = int((self.audioRF_history + self.audioRF_future) / self.fps * self.sample_rate)
        
         
        if self.task == 'Audio2Feature':
            self.seq_len = cfg.sequence_length
            if cfg.feature_encoder == 'WaveNet':
                self.A2L_receptive_field = cfg.A2L_receptive_field
                self.A2L_item_length = self.A2L_receptive_field + self.target_length - 1
            elif cfg.feature_encoder == 'LSTM':
                self.A2L_receptive_field = 30
                self.A2L_item_length = self.A2L_receptive_field + self.target_length - 1
            else:
                pass
        elif self.task == 'Audio2Headpose':
            self.A2H_receptive_field = cfg.A2H_receptive_field
            self.A2H_item_length = self.A2H_receptive_field + self.target_length - 1
            self.audio_window = cfg.audio_windows
            self.half_audio_win = int(self.audio_window / 2)
            
        self.frame_future = cfg.frame_future
        self.predict_length = cfg.predict_length
        self.predict_len = int(self.predict_length / 2)
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.mouth_related_indices = np.concatenate([np.arange(4, 13), np.arange(48, 68)])  # 68, 29*3
        if self.task == 'Audio2Feature':
            if cfg.only_mouth:
                self.indices = self.mouth_related_indices
            else:
                self.indices = np.arange(68)
        if cfg.use_delta_pts:
            self.pts3d_mean = np.load(os.path.join(self.data3d_label_dir, 'mean_pts_3d_fixed_countour_nose_closemouth.npy'))
        
        self.audio, _ = librosa.load(self.audio_file, sr=self.sample_rate)
        
        if cfg.audio_encoder == 'APC':
            APC_feature_file = '{}_APC_feature_{}.npy'.format(cfg.APC_audio_person, cfg.APC_audio_person_version)
            APC_feature_path = os.path.join(self.APC_dir, APC_feature_file)
            need_deepfeatures = False if os.path.exists(APC_feature_path) else True
            need_deepfeatures = True if cfg.re_audio_encoder else False
            if not need_deepfeatures:
                self.audio_features = np.load(APC_feature_path).astype(np.float32)
        else:
            need_deepfeatures = False
            
        if self.task == 'Audio2Feature':
            self.start_point = 0
        elif self.task == 'Audio2Headpose':
            self.start_point = 300
        
        self.fit_data = np.load(os.path.join(self.data3d_label_dir, '3d_fit_data.npz'))
        self.frontlized_data = np.load(os.path.join(self.data3d_label_dir, 'landmarks3d_fixed_countour_nose.npy'))   # change to fix data 
        if cfg.use_delta_pts:
            # self.pts3d = self.fit_data['pts_3d'] - self.pts3d_mean  # pts3d_diff
            self.pts3d = self.frontlized_data - self.pts3d_mean
        else:
            # self.pts3d = self.fit_data['pts_3d']
            self.pts3d = self.frontlized_data
        
        self.feats = self.pts3d.squeeze()   # for dece save 
        
        self.rot_angles = self.fit_data['rot_angles'].astype(np.float32)
        
        # may handle your rot_angles 
        
        # use delta translation 
        self.mean_trans = self.fit_data['trans'][..., 0].astype(np.float32).mean(axis=0)
        self.trans = self.fit_data['trans'][..., 0] - self.mean_trans
        
        self.headposes = np.concatenate([self.rot_angles, self.trans], axis=1)
        self.velocity_pose = np.concatenate([np.zeros(6)[np.newaxis, :], self.headposes[1:] - self.headposes[:-1]])
        self.acceleration_pose = np.concatenate([np.zeros(6)[np.newaxis,:], self.velocity_pose[1:] - self.velocity_pose[:-1]])
        
        total_frames = self.feats.shape[0] - 60   # TODO ?
        
        if need_deepfeatures and cfg.audio_encoder == 'APC':
            mel80 = compute_mel_one_sequence(self.audio)
            mel_nframe = mel80.shape[0]
            
            APC_model = APC_encoder(cfg.audiofeature_input_channels,
                                    cfg.APC_hidden_size,
                                    cfg.APC_rnn_layers,
                                    cfg.APC_residual)
            APC_model.load_state_dict(torch.load(self.APC_model_path, map_location=str(self.device)), strict=False)
            APC_model.to(self.device)
            APC_model.eval()
            
            with torch.no_grad():
                length = torch.Tensor([mel_nframe])
                mel80_torch = torch.from_numpy(mel80.astype(np.float32)).to(self.device).unsqueeze(0)
                hidden_reps = APC_model.forward(mel80_torch, length)[0]
                hidden_reps = hidden_reps.cpu().numpy()
                np.save(APC_feature_path, hidden_reps)
                self.audio_features = hidden_reps
                
        valid_frames = total_frames - self.start_point
        self.len = valid_frames - 400  # TODO ?
        self.sample_start = [0]
        
        self.total_len = np.int32(np.floor(self.len) / self.frame_jump_stride)
        
    def __getitem__(self, index):
        index_real = np.int32(index * self.frame_jump_stride)
        current_frame = index_real + self.start_point
        current_target_length = self.target_length  # for headpose
        
        if self.task == 'Audio2Feature':
            A2Lsamples = self.audio_features[current_frame * 2: (current_frame + self.seq_len) * 2]
            target_pts3d = self.feats[current_frame: current_frame + self.seq_len, self.indices].reshape(self.seq_len, -1) 
            
            A2Lsamples = torch.from_numpy(A2Lsamples).float()
            target_pts3d = torch.from_numpy(target_pts3d).float()
            
            return A2Lsamples, target_pts3d
        
        elif self.task == 'Audio2Headpose':
            if self.half_audio_win == 1:
                A2H_history_start = current_frame - self.A2H_receptive_field
                A2Hsamples = self.audio_features[2*(A2H_history_start + self.frame_future): 2*(A2H_history_start+self.frame_future+self.A2H_item_length)]
            else:
                A2Hsamples = np.zeros([self.A2H_item_length, self.audio_window, 512])
                for i in range(self.A2H_item_length):
                    A2Hsamples[i] = self.audio_features[2 * (A2H_history_start + i) - self.half_audio_win : 2 * (A2H_history_start + i) + self.half_audio_win]
            
            if self.predict_len == 0:
                target_headpose = self.headposes[A2H_history_start + self.A2H_receptive_field : A2H_history_start + self.A2H_item_length + 1]
                history_headpose = self.headposes[A2H_history_start : A2H_history_start + self.A2H_item_length].reshape(self.A2H_item_length, -1)
                
                target_velocity = self.velocity_pose[A2H_history_start + self.A2H_receptive_field : A2H_history_start + self.A2H_item_length + 1]
                history_velocity = self.velocity_pose[A2H_history_start : A2H_history_start + self.A2H_item_length].reshape(self.A2H_item_length, -1)
                target_info = torch.from_numpy(np.concatenate([target_headpose, target_velocity], axis=1).reshape(current_target_length, -1)).float()
            else:
                history_headpose = self.headposes[A2H_history_start:A2H_history_start+self.A2H_item_length].reshape(self.A2H_item_length, -1)
                history_velocity = self.velocity_pose[A2H_history_start:A2H_history_start+self.A2H_item_length].reshape(self.A2H_item_length, -1)
                
                target_headpose_ = self.headposes[A2H_history_start + self.A2H_receptive_field - self.predict_len: A2H_history_start + self.A2H_item_length + self.predict_len + 2]
                target_headpose = np.zeros([current_target_length, self.predict_length, target_headpose_.shape[1]])
                
                for i in range(current_target_length):
                    target_headpose[i] = target_headpose_[i : i + self.predict_length]
                
                target_velocity_ = self.headposes[A2H_history_start + self.A2H_receptive_field - self.predict_len : A2H_history_start + self.A2H_item_length + 1 + self.predict_len + 1]
                target_velocity = np.zeros([current_target_length, self.predict_length, target_velocity_.shape[1]])
                
                for i in range(current_target_length):
                    target_velocity[i] = target_velocity_[i : i + self.predict_length]
            
                target_info = torch.from_numpy(np.concatenate([target_headpose, target_velocity], axis=2).reshape(current_target_length, -1)).float()
                target_info = target_info[:, :12] 
            
            A2Hsamples = torch.from_numpy(A2Hsamples).float()
            history_info = torch.from_numpy(np.concatenate([history_headpose, history_velocity], axis=1)).float()
            
            return A2Hsamples, history_info, target_info


    def __len__(self):
        return self.total_len
