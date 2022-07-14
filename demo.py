from lib.config.config import Config
from lib.lsp import LSP

import torch 
import librosa
import os
import numpy as np

from tqdm import tqdm
import cv2
from lib.models.networks import APC_encoder
from lib.utils.visisual import plot_kpts

from lib.audio_module.utils import compute_mel_one_sequence, KNN_with_torch, compute_LLE_projection_all_frame


def ndc2img(landmarks3d):
    
    landmarks3d[...,0] = landmarks3d[...,0]*256 + 256
    landmarks3d[...,1] = landmarks3d[...,1]*256 + 256
    return landmarks3d[..., :2]


def infer(cfg):
    h, w, sr, FPS = 512, 512, 16000, 60
    mouth_indices = np.concatenate([np.arange(4, 13), np.arange(48, 68)])  # 9+20
    lsp = LSP(cfg)
    
    label_dir = os.path.join(cfg.data_dir, cfg.d3label_dir)
    apc_feature_base = os.path.join(cfg.data_dir, cfg.audio_encoder, 'who_say_APC_feature_who_say_vx.npy')
    
    
    audio, _ = librosa.load(cfg.test_audio, sr=sr)
    total_frames = np.int32(audio.shape[0] / sr * FPS)
    # Compute APC feature
    mel80 = compute_mel_one_sequence(audio, device='cuda') # (x, 80)
    mel_nframe = mel80.shape[0]
    
    APC_model = APC_encoder(cfg.APC_infer.mel_dim, cfg.APC_infer.hidden_size,
                            cfg.APC_infer.num_layers, cfg.APC_infer.residual)
    APC_model.load_state_dict(torch.load(cfg.APC_infer.ckp_path), strict=False)
    APC_model.to('cuda')
    APC_model.eval()
    
    with torch.no_grad():
        length = torch.Tensor([mel_nframe])
        mel80_torch = torch.from_numpy(mel80.astype(np.float32)).to('cuda').unsqueeze(0)  # (1, x, 80)
        hidden_reps = APC_model.forward(mel80_torch, length)[0]   # [mel_nframe, 512]， [x 512]
        hidden_reps = hidden_reps.cpu().numpy()
    audio_feats = hidden_reps
    
    APC_feat_database = np.load(apc_feature_base, allow_pickle=True)   # (x, 512), FPS 60, Knear 10 is cp from V0324**
    if cfg.use_LLE:
        print('2. Manifold projection...')
        ind = KNN_with_torch(audio_feats, APC_feat_database, K=cfg.APC_infer.Knear)
        weights, feat_fuse = compute_LLE_projection_all_frame(audio_feats, APC_feat_database, ind, audio_feats.shape[0])
        audio_feats = audio_feats * (1-cfg.APC_infer.LLE_percent) + feat_fuse * cfg.APC_infer.LLE_percent  # [x 512]
    
    pred_Feat = lsp.inference(audio_feats)
    nframe = pred_Feat.shape[0]
    pred_pts3d = np.zeros([nframe, 68, 3])
    pred_pts3d[:, mouth_indices] = pred_Feat.reshape(-1, 29, 3)[:nframe]  # only_mouth, may do some smooth or re-correct
    mean_pts3d = np.load(os.path.join(label_dir, 'mean_pts_3d.npy'))   # (68, 3)
    pred_pts3d = pred_pts3d + mean_pts3d  # remove mouth value of mean_pts3d 
    
    # compute projected landmarks
    # pred_landmarks = np.zeros([nframe, 68, 2], dtype=np.float32)
    final_pts3d = np.zeros([nframe, 68, 3], dtype=np.float32)
    # final_pts3d[:] = std_mean_pts3d.copy() 
    final_pts3d[:] = mean_pts3d.copy()
    final_pts3d[:, 48:68] = pred_pts3d[:nframe, 48:68]
    pred_landmarks = ndc2img(final_pts3d)
    
    img_ = np.zeros((512, 512), dtype=np.uint8)
    
    for ind in tqdm(range(0, nframe), desc='Image2Image translation inference'):
        img_edges = plot_kpts(img_, pred_landmarks[ind])
        cv2.imwrite(os.path.join(cfg.data_dir,'test_mouth', f'mouth_{ind}.png'), img_edges)


if __name__ == "__main__":
    path = './configs/audio2feature.yaml'
    cfg = Config.fromfile(path)
    infer(cfg)
    # img2video