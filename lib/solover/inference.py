import torch 
import numpy as np
from lib.models.audio2feature import Audio2Feature



def a2f_inference(audio_feats, cfg):
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
    a2f = Audio2Feature(cfg)   # load weights
    # torch.load()

    frame_future = cfg.frame_future
    nframe = int(audio_feats.shape[0] / 2) 
    
    if not frame_future == 0:
        audio_feats_insert = np.repeat(audio_feats[-1],  2 * (frame_future)).reshape(-1, 2 * (frame_future)).T
        audio_feats = np.concatenate([audio_feats, audio_feats_insert])

    with torch.no_grad():
        input = torch.from_numpy(audio_feats).unsqueeze(0).float().to('cuda')
        preds = a2f(input)
        
        # drop first frame future results
    if not frame_future == 0:
        preds = preds[0, frame_future:].cpu().detach().numpy()
    else:
        preds = preds[0, :].cpu().detach().numpy()
    
    assert preds.shape[0] == nframe
    return preds


def a2h_inference(audio_feats, cfg):
    pass
