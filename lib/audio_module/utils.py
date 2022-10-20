import torch
import numpy as np
from tqdm import tqdm
from numpy.linalg import solve
from .audio_funcs import Audio2Mel


def compute_mel_one_sequence(audio, hop_length=16000//120, winlen=1/60, winstep=0.5/60, sr=16000, fps=60, device='cuda'):
    '''
    compute mel for an audio sequence, winlen is wrong ?
    '''    
    device = torch.device(device)
    nframe = int(audio.shape[0] / sr * fps)
    mel_nframe = 2 * nframe
    mel_frame_len = int(sr * winlen)  # int(1600/60)
    mel_frame_step = int(sr * winstep)
    mel80s = np.zeros([mel_nframe, 80])
    
    # Audio2Mel_torch = Audio2Mel(n_fft=512, hop_length=hop_length, win_length=mel_frame_len, sampling_rate=sr, n_mel_channels=80).to(device)
    Audio2Mel_torch = Audio2Mel(n_fft=512, hop_length=mel_frame_len, win_length=hop_length, sampling_rate=sr, n_mel_channels=80).to(device)
    
    
    for i in range(mel_nframe):
        st = i * mel_frame_step 
        audio_clip = audio[st:st+mel_frame_len]
        if len(audio_clip) < mel_frame_len:
            audio_clip = np.concatenate([audio_clip, np.zeros([mel_frame_len - len(audio_clip)])])
        audio_clip_device = torch.from_numpy(audio_clip).unsqueeze(0).unsqueeze(0).to(device).float()
        
        mel80s[i] = Audio2Mel_torch(audio_clip_device).cpu().numpy()[0].T    # [1, 80]
        
    return mel80s


def KNN_with_torch(feats, feat_database, K=10):
    feats = torch.from_numpy(feats)  #.cuda()
    feat_database = torch.from_numpy(feat_database)  #.cuda()
    # Training
    feat_base_norm = (feat_database ** 2).sum(-1)
#    print('start computing KNN ...')
#    st = time.time()      
    feats_norm = (feats ** 2).sum(-1)
    diss = (feats_norm.view(-1, 1)
            + feat_base_norm.view(1, -1)
            - 2 * feats @ feat_database.t()  # Rely on cuBLAS for better performance!
        )
    ind = diss.topk(K, dim=1, largest=False).indices
#    et = time.time()
#    print('Taken time: ', et-st)
    
    return ind.cpu().numpy()


def solve_LLE_projection(feat, feat_base):
    '''find LLE projection weights given feat base and target feat
    Args:
        feat: [ndim, ] target feat
        feat_base: [K, ndim] K-nearest feat base
    =======================================
    We need to solve the following function
    ```
        min|| feat - \sum_0^k{w_i} * feat_base_i ||, s.t. \sum_0^k{w_i}=1
    ```
    equals to:
        ft = w1*f1 + w2*f2 + ... + wk*fk, s.t. w1+w2+...+wk=1
           = (1-w2-...-wk)*f1 + w2*f2 + ... + wk*fk
     ft-f1 = w2*(f2-f1) + w3*(f3-f1) + ... + wk*(fk-f1)
     ft-f1 = (f2-f1, f3-f1, ..., fk-f1) dot (w2, w3, ..., wk).T
        B  = A dot w_,  here, B: [ndim,]  A: [ndim, k-1], w_: [k-1,]
    Finally,
       ft' = (1-w2-..wk, w2, ..., wk) dot (f1, f2, ..., fk)
    =======================================    
    Returns:
        w: [K,] linear weights, sums to 1
        ft': [ndim,] reconstructed feats
    '''
    K, ndim = feat_base.shape
    if K == 1:
        feat_fuse = feat_base[0]
        w = np.array([1])
    else:
        w = np.zeros(K)
        B = feat - feat_base[0]   # [ndim,]
        A = (feat_base[1:] - feat_base[0]).T   # [ndim, K-1]
        AT = A.T
        w[1:] = solve(AT.dot(A), AT.dot(B))
        w[0] = 1 - w[1:].sum()
        feat_fuse = w.dot(feat_base)
      
    return w, feat_fuse


def compute_LLE_projection_all_frame(feats, feat_database, ind, nframe):
    nframe = feats.shape[0]
    feat_fuse = np.zeros_like(feats)
    w = np.zeros([nframe, ind.shape[1]])
    for i in tqdm(range(nframe), desc='LLE projection'):
        current_K_feats = feat_database[ind[i]]  # (10, 512)
        w[i], feat_fuse[i] = solve_LLE_projection(feats[i], current_K_feats)  # feats[i]:(512,)
    
    return w, feat_fuse


