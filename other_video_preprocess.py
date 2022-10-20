import numpy as np
import cv2
from os.path import join as pjoin
from lib.utils.util import read_videos
from scipy.spatial.transform import Rotation
from lib.datasets.face_utils import rigid_transform_3D

def openrate(lmark1):
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    open_rate1 = []
    for k in range(3):
        open_rate1.append(np.absolute(
            lmark1[open_pair[k][0], :2] - lmark1[open_pair[k][1], :2]))

    open_rate1 = np.asarray(open_rate1)
    return open_rate1.mean()

def get_best_front_frame(landmarks):
    '''
    return the best 
    '''
    pass


def RT_compute(landmarks, save_dir=None):
    landmarks = np.squeeze(landmarks, axis=1)
    consider_key = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 27, 28, 29, 30, 31,
                    32, 33, 34, 35, 39, 42, 36, 45, 17, 21, 22, 26]

    source = landmarks[0][consider_key, ...]  # first frame or the best front frame
    
    source = np.mat(source)
    
    landmarks_part = landmarks[:,consider_key, :]

    num_landmark = landmarks.shape[0]
    RTs = np.zeros((num_landmark, 6))
    frontlized = np.zeros((num_landmark, 68, 3))
    
    for i in range(num_landmark):
        
        target = np.mat(landmarks_part[i])
        ret_R, ret_t = rigid_transform_3D(target, source)
        source_lmark = np.mat(landmarks[i])
        A2 = ret_R * source_lmark.T
        A2 += np.tile(ret_t, (1, 68))
        A2 = A2.T
        frontlized[i] = A2
        r = Rotation.from_matrix(ret_R)  # from_dcm  deprecated! 1.4.0->1.6.0
        vec = r.as_rotvec()
        RTs[i, :3] = vec
        RTs[i, 3:] = np.squeeze(np.asarray(ret_t))
        
    if save_dir is not None:
        print(save_dir)
        np.save(pjoin(save_dir, 'RT.npy'), RTs)
        np.save(pjoin(save_dir, 'frontlized.npy'), frontlized)
        fix_mean = pjoin(save_dir, 'mean_pts_3d.npy')
        np.save(fix_mean, np.array(frontlized).mean(axis=0))

def get_front_video(video_path):
    v_frames = read_videos(video_path)
    rt_path = video_path[:-4] + '__rt.npy'
    rt = np.load(rt_path)
    lmark_length = rt.shape[0]
    find_rt = []
    for t in range(0, lmark_length):
        find_rt.append(sum(np.absolute(rt[t, :3])))
    find_rt = np.asarray(find_rt)

    min_index = np.argmin(find_rt)

    img_path = video_path[:-4] + '__%05d.png' % min_index

    print('save it to ' + img_path)
    cv2.imwrite(img_path, v_frames[min_index])
    
def merge_audio_landmark():
    'ffmpeg -i "concat:./data/hk_fake_8_14/video_audio/audio.wav|./data/hk_fake_38/video_audio/audio.wav" -acodec copy  merge_audio.wav'
    import librosa
    a1f = './data/hk_fake_8_14/video_audio/audio.wav'
    a2f = './data/hk_fake_38/video_audio/audio.wav'
    a1, _ = librosa.load(a1f, sr=16000)
    a2, _ = librosa.load(a2f, sr=16000)
    
    merge_audio = np.hstack([a1, a2])
    print(merge_audio.shape, a1.shape, a2.shape)
    librosa.output.write_wav('merge_audio_librosa.wav', merge_audio, sr=16000)

    
if __name__ == "__main__":
    
    base_dir = './data/hk_fake_8_14/label'
    base_dir_ = './data/hk_fake_38/label'
    # data_fit = np.load(pjoin(base_dir, '3d_fit_data.npz'))
    # landmarks = data_fit['pts_3d']
    # RT_compute(landmarks, save_dir=base_dir)
    print('test')
    # x = np.load('./data/hk_fake_38/label/RT.npy')
    # print(x.shape, x[0], x[1])
    
    frontlized = np.load('./data/hk_fake_8_14/label/frontlized.npy')
    mean_f = np.array(frontlized).mean(axis=0)
    # mean_f[:, 1] -=5
    fix_mean = './data/hk_fake_8_14/label/mean_pts_3d.npy'
    np.save(fix_mean, mean_f)
    
    # f1 = np.load(pjoin(base_dir, 'frontlized.npy'))
    # f2 = np.load(pjoin(base_dir_, 'frontlized.npy'))
    # f_merge = np.concatenate((f1, f2), axis=0)
    # np.save('frontlized_merge.npy', f_merge)
    # print(f_merge.shape)
    
    # merge_audio_landmark()
    
    