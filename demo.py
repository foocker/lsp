from turtle import window_height
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
from lib.utils.postprocess import enhnace_mouth

from skimage.io import imread
import albumentations.pytorch as alp
from lib.utils.visisual import draw_face_feature_maps

from lib.utils.util import save_image, tensor2im

from lib.datasets.face_dataset import FaceDatasetCustom
from lib.datasets.face_utils import mse_metrix, openrate, mounth_open2close, eye_blinking_o, eye_blinking_inverse, smooth, oned_smooth
from skimage.transform import estimate_transform

from lib.utils.postprocess import solve_intersect_mouth, mouth_pts_AMP, landmark_smooth_3d, headpose_smooth
from data_analysis import angle2matrix, SmoothScatterPoints

def ndc2img(landmarks3d):
    
    landmarks3d[...,0] = landmarks3d[...,0]*256 + 256
    landmarks3d[...,1] = landmarks3d[...,1]*256 + 256
    
    return landmarks3d[..., :2]


def img2video(image_folder, video_name, img_format='png', fps=60.0):
    # f = lambda x: float(x.split('_')[-1][:-4])
    f = lambda x: float(x.split('_')[1])
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".{}".format(img_format))]
    images = sorted(images, key=f)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4v')
    # fourcc = 0x7634706d

    video = cv2.VideoWriter(video_name, 0x7634706d, fps, (width, height), True)

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        video.write(img)

    video.release()

    print("Succeeds!")
    
    
def get_candidates(p_dir):
    img_candidates = []
    imgs = os.listdir(p_dir)
    for iname in imgs:
        img = imread(os.path.join(p_dir, iname))
        transform = alp.transforms.ToTensor(normalize={'mean':(0.5, 0.5, 0.5), 'std':(0.5, 0.5, 0.5)})
        img_tensor = transform(image=img)['image']
        img_candidates.append(img_tensor)
    img_candidates = torch.cat(img_candidates).unsqueeze(0)
    
    return img_candidates


def infer(cfg):
    h, w, sr, FPS = 512, 512, 16000, 60
    # mouth_indices = np.concatenate([np.arange(4, 13), np.arange(48, 68)]) if cfg.output_size == 29 else np.arange(68)  # 9+20
    enhance_mouth_index = np.arange(48, 68)
    mouth_indices = np.arange(48, 68)  # 20 if countour is bad, please optimazation the model
    mouth_inner_indices = np.arange(60, 68)
    
    ## headpose 
    data_label_dir = os.path.join(cfg.data_dir, cfg.d3label_dir)
    fit_data = np.load(os.path.join(data_label_dir, '3d_fit_data.npz'))
    landmarks_3d_fixed = np.load(os.path.join(data_label_dir, 'landmarks3d_fixed_countour_nose.npy')) # frontlized.npy, landmarks3d_fixed_countour_nose.npy
    landmark_3d_fix_mean = np.load(os.path.join(data_label_dir, 'mean_pts_3d.npy'))  # mean_pts_3d_fixed_countour_nose_closemouth, mean_pts_3d
    pts3d_sub_mean = landmarks_3d_fixed - landmark_3d_fix_mean
    
    eye_brow_indices = np.array(list(range(17, 27)), np.int32)
    candidate_eye_brow = pts3d_sub_mean[10:, eye_brow_indices]
    
    mean_trans = fit_data['trans'][..., 0].astype(np.float32).mean(axis=0)
    
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
    
    pred_feat = lsp.inference(audio_feats)
    
    nframe = pred_feat.shape[0]
    
    ## mouth
    need_feat = pred_feat.reshape(-1, cfg.output_size//3, 3)[:nframe]  # 效果不明显, TODO 
    pred_pts3d = np.zeros([nframe, 68, 3])
    pred_pts3d[:, mouth_indices] = need_feat[:, mouth_indices]  # only_mouth, may do some smooth or re-correct
    
    mean_pts3d = np.load(os.path.join(label_dir, 'mean_pts_3d.npy'))   # (68, 3), mean_pts_3d, mean_pts_3d_fixed_countour_nose_closemouth
    
    # xmean = draw_face_feature_maps(ndc2img(mean_pts3d))
    # cv2.imwrite('gxxmean.png', xmean)  # right
    
    if cfg.test_mode.close_mean_pts3d_mouth:
        op = openrate(mean_pts3d*256)  # -3.3567418124602617
        print(f'mean_pts3d openrate is: {op}')
        if abs(op) > 1.8:
            mean_pts3d = mounth_open2close(mean_pts3d)
    
    if cfg.test_mode.use_fix_contour:
        pred_pts3d[:, range(17), ...] = mean_pts3d[range(17), ...]   # fix contour
        
    if cfg.test_mode.only_pred:
        pred_pts3d = pred_pts3d + mean_pts3d  # this means should fix the contour
        pred_landmarks = ndc2img(pred_pts3d)
    
    if cfg.test_mode.mouth_smooth:
        pred_pts3d = landmark_smooth_3d(pred_pts3d, cfg.Feat_smooth_sigma, area='only_mouth')
        SSP = SmoothScatterPoints()
        methods = [SSP.SG, SSP.move_avg]  # move_avg one dimension
        # parameters = [(5, 3, {'axis':1}), 5]
        method = methods[0]
        mouth_x = method(pred_pts3d[:, mouth_indices, 0], 5, 3, axis=-1)  # mouth_inner_indices, mouth_indices
        mouth_y = method(pred_pts3d[:, mouth_indices, 1], 5, 3, axis=-1)
        mouth_z = method(pred_pts3d[:, mouth_indices, 2], 5, 3, axis=-1)
        mouth_xyz = np.concatenate([mouth_x[...,np.newaxis], mouth_y[...,np.newaxis], mouth_z[...,np.newaxis]], axis=-1)
        # pred_pts3d = mouth_pts_AMP(pred_pts3d, True, cfg.AMP_method, cfg.Feat_AMPs)  # test Feat_AMPs[1] for enhance y of mouth TODO
        pred_pts3d[:, mouth_indices] = mouth_xyz
        pred_pts3d = pred_pts3d + mean_pts3d
        # pred_pts3d = solve_intersect_mouth(pred_pts3d)
        
    if cfg.test_mode.test_mouth_openrate:
        for i in range(nframe):
            op = openrate(pred_pts3d[i]*256)
            if abs(op) < 1.8:
                print(i, op)

    if cfg.test_mode.pred_normal:
        # TODO 逻辑优化 
        pred_landmarks = np.zeros([nframe, 68, 2], dtype=np.float32)
        final_pts3d = np.zeros([nframe, 68, 3], dtype=np.float32)
        final_pts3d[:] = mean_pts3d.copy()
        if cfg.test_mode.mouth_smooth:
            final_pts3d[:, mouth_indices] = pred_pts3d[:nframe, mouth_indices]
        else:
            final_pts3d[:, mouth_indices] += pred_pts3d[:nframe, mouth_indices]
        
        pred_landmarks = ndc2img(final_pts3d)
        # pred_landmarks = smooth(pred_landmarks)  # 前后alpha linear combine
        if cfg.test_mode.open_eye:
            pred_landmarks = eye_blinking_inverse(pred_landmarks, fps=60, keep_second=5) 
            # 增加眼睛动起来模块 TODO 加强 嘴巴的运动幅度， enhnace_mouth
            # for i in range(pred_landmarks.shape[0]):
            #     img_edge = draw_face_feature_maps(pred_landmarks[i])
            #     cv2.imwrite(f'./temp_test/{i}.png', img_edge)
        if cfg.test_mode.enhance_mouth:
            pass
        
    if cfg.test_mode.headpose:
        print('Headpose inference...')
        # set history headposes as zero
        pre_headpose = np.zeros(cfg.A2H_wavenet_input_channels, np.float32)  # (12,)
        pred_Head = lsp.inference_h(audio_feats, pre_headpose, fill_zero=True, sigma_scale=0.3, cfg=cfg)
        
        nframe = min(pred_feat.shape[0], pred_Head.shape[0])
        
        pred_Head[:, 0:3] *= cfg.Headpose.AMP[0]   # AMP=1
        pred_Head[:, 3:6] *= cfg.Headpose.AMP[1]  #  trans_AMP:1

        pred_headpose = headpose_smooth(pred_Head[:,:6], cfg.Headpose.smooth).astype(np.float32)
        pred_headpose[:, 3:] += mean_trans
        # pred_headpose[:, 0] += 180  # (-pi, pi)+shift, 头部旋转 x轴 + 180
        
        # ------- projection landmarks ---------
        final_pts3d = np.zeros([nframe, 68, 3], dtype=np.float32)
        final_pts3d[:] = mean_pts3d.copy()  
        final_pts3d[:, 48:68] = pred_pts3d[:nframe, 48:68]  # mouth
        
        for k in tqdm(range(nframe)):
            ind = k % candidate_eye_brow.shape[0]  # 眼睛，眉毛 偏移
            final_pts3d[k, eye_brow_indices] = candidate_eye_brow[ind] + mean_pts3d[eye_brow_indices]
            angele_xyz = pred_headpose[k][:3]
            R = angle2matrix(angele_xyz)
            T = pred_headpose[k][3:][np.newaxis, :]
            final_pts3d[k] = (R @ final_pts3d[k].T + T.T).T
            
        pred_landmarks = ndc2img(final_pts3d)  # headpose
        # ------- projection landmarks ---------
    
    # -------- close open when not speeaking ------------
    for i in range(nframe):
        opi = openrate(pred_landmarks[i])
        if abs(opi) < 1.8:
            pred_landmarks[i] = mounth_open2close(pred_landmarks[i])
    # -------- close open when not speeaking  ------------

    img_candidatates = get_candidates(os.path.join(cfg.data_dir, cfg.candidates_dir)).to(lsp.device)
    
    if not os.path.exists(cfg.test_all_dir):
        os.makedirs(cfg.test_all_dir, exist_ok=True)
        
    if not os.path.exists(cfg.infer_dir):
        os.makedirs(cfg.infer_dir, exist_ok=True)
    
    if cfg.enhance_mouth_scale > 1:
        for i in range(nframe):
            pred_landmarks[i] = enhnace_mouth(pred_landmarks[i], cfg.enhance_mouth_scale)
    
    if cfg.test_mode.traindata_mean_shift:
        FD = FaceDatasetCustom(cfg)
        shift_landmarks = list(range(0, 17)) + list(range(27,36))
        pred_landmarks_mean = ndc2img(mean_pts3d)
        src = pred_landmarks_mean[shift_landmarks, ...]

    if (cfg.test_mode.pred_mouth_shift_by_traindata and not cfg.test_mode.headpose) or cfg.test_mode.pred_aligne_traindata:
        # ----- add shift by orignal------
        pts2d = np.load(os.path.join(label_dir, 'landmarks2d_original.npy'))
        pts2d = np.squeeze(pts2d)
        pts2d = pts2d * 256 + 256
        
        nose = list(range(27, 36))
        contour = list(range(0, 27))
        mouth = list(range(48, 68))
        
        similarity = nose + contour
        src_pts = pred_landmarks[:, similarity]
        dst_pts = pts2d[:, similarity]
        keep_mouth = [48, 60, 54, 64]
    
    if not os.path.exists(os.path.join(cfg.data_dir,'test_mouth')):
        os.makedirs(os.path.join(cfg.data_dir,'test_mouth'), exist_ok=True)
        
    # ----- use trian landmark but replace mouth by pred --------
    for ind in tqdm(range(1, nframe), desc='Image2Image translation inference'):
        # TODO 换种写法
        if cfg.test_mode.traindata_mean_shift:  
            # ----- add shift------
            data_frame = FD[ind]
            dst = data_frame['points'][shift_landmarks, ...]
            diff_xy = (dst - src).mean(0)
            if ind == 100:
                print(diff_xy)
            shifted_mean_pts2d = pred_landmarks[ind] + diff_xy
            img_edges = draw_face_feature_maps(shifted_mean_pts2d)  # 预测值平移平均landmark到原始数据之间的偏移量
            # ----- add shift------
            
        elif cfg.test_mode.pred_aligne_traindata:
            # 和 traindata_mean_shift 差别不大, 对预测做到训练集的平移和旋转，多了旋转
            tform = estimate_transform('affine', src_pts[ind], dst_pts[ind])
            R = tform.params[:2, :2]
            T = tform.params[:2, 2]
            transformed_pts = pred_landmarks[ind]@R + T
            img_edges = draw_face_feature_maps(transformed_pts)  # 原始预测变形到对齐训练集
            
        elif cfg.test_mode.pred_mouth_shift_by_traindata and not cfg.test_mode.headpose:
            # 和 traindata_mean_shift不同，使用被平移的预测嘴巴，替换原始的嘴部信息
            tform = estimate_transform('affine', src_pts[ind], dst_pts[ind])  # euclidean, similarity, affine, piecewise-affine, polynomial
            R = tform.params[:2, :2]
            T = tform.params[:2, 2]
            transformed_pts = pred_landmarks[ind]@R + T
            mouth_fixed_locate = pts2d[ind][keep_mouth]
            mouth_move_locate = transformed_pts[keep_mouth]
            mouth_shift = mouth_fixed_locate - mouth_move_locate
            
            shift_x = mouth_shift[:, 0].mean()
            shift_y = mouth_shift[:, 1].mean()
            
            pts2d[ind][mouth] = transformed_pts[mouth] + np.array([shift_x, shift_y])  # 相对平移
            img_edges = draw_face_feature_maps(pts2d[ind])   # 训练集被替换了预测的嘴部信息
            
        elif cfg.test_mode.traindata_trainaudio:
            assert cfg.test_audio.split('/')[-1] == 'audio.wav'
            if ind == 1:
                print('prediction by the training audio')
            img_edges = draw_face_feature_maps(pred_landmarks[ind])  # 测试训练模型在训练数据集上的生成效果
        
        elif cfg.test_mode.pred_normal:
            if ind == 1:
                print('prediction on audio', cfg.test_audio.split('/')[-1])
            img_edges = draw_face_feature_maps(pred_landmarks[ind])  # 原始预测，若有headpose则待运动，否则就是嘴部和眼睛的运动
            
        else:
            raise ValueError('test mode not contained!')
        
        if cfg.test_mode.save_edge:
            cv2.imwrite(os.path.join(cfg.data_dir,'test_mouth', f'mouth_{ind}_m.png'), img_edges)

        if not cfg.test_mode.only_edge:
            feature_map = img_edges[np.newaxis, :].astype(np.float32)/255.
            feature_map = torch.from_numpy(feature_map).unsqueeze(0).to(lsp.device)
            fake_pred = lsp.inference_g(feature_map, img_candidatates)
            pred_fake = tensor2im(fake_pred[0])
            save_image(pred_fake, os.path.join(cfg.infer_dir, f'infer_{ind}_temp.png'))
            
        # if ind == 1200:
        #     break
        
    
    if cfg.test_mode.only_edge:
        savevideo_name = os.path.join(cfg.data_dir,'infer_result_mouth_edge.mp4')

        img2video(os.path.join(cfg.data_dir,'test_mouth'), savevideo_name)
    else:
        
        savevideo_name = os.path.join(cfg.data_dir,'infer_result.mp4')
        img2video(cfg.infer_dir, savevideo_name)
        
        if cfg.rm_infer_temp:
            import shutil
            try:
                print(f'rm {cfg.infer_dir}')
                shutil.rmtree(cfg.infer_dir)
            except OSError as e:
                print(f"Error: {cfg.infer_dir}: {e.strerror}")
                
        # merge audio 
        from video_preprocess import FAN
        
        fan = FAN()
        fan.merge_audio_video(savevideo_name, cfg.test_audio)
    
# TODO add audio, speed


if __name__ == "__main__":
    import time
    path = './configs/audio2feature.yaml'
    st = time.time()
    cfg = Config.fromfile(path)
    infer(cfg)
    end_time = time.time()
    print('cost time is:', end_time - st)