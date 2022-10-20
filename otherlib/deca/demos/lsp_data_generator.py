import os, sys


import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch
import subprocess
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

from decalib.utils.rotation_converter import batch_euler2axis


h, w, sr, FPS = 512, 512, 16000, 60

def get_change_paras():
    '''
     As illustrated in the paper, I tracked the face on the original resolution. However, 
     our desired results need to cut&crop&resize the image, therefore, 'scale, xc, yc' and 
     other parameters denote the cut&crop&resize parameters. I need these parameters to transform 
     the tracking results& camera parameters to fit the desired results (that is about resolution, 
     face location, etc.)
    0.7954545454545454 754 278
    scale: resize
    xc: cut&crop
    yc: cut&crop
    '''
    pass

def get_3d_fit_data():
    '''
    see DECA 
    pts_3d:(13369, 73, 3), headpose, landmark, 同时得到mean_pts3d.npy
    rot_angles: (13369, 3), ? TODO
    trans: (13369, 3, 1), trans
    '''
    np.save('3d_fit_data.npz', None)
    pass

def get_camera_intrinsic():
    '''
    参考:https://towardsdatascience.com/camera-calibration-with-example-in-python-5147e945cdeb
    相机标定
    常用术语
    内参矩阵: Intrinsic Matrix
    焦距: Focal Length
    主点: Principal Point
    径向畸变: Radial Distortion
    切向畸变: Tangential Distortion
    旋转矩阵: Rotation Matrices
    平移向量: Translation Vectors
    平均重投影误差: Mean Reprojection Error
    重投影误差: Reprojection Errors
    重投影点: Reprojected Points
    https://blog.csdn.net/lql0716/article/details/71973318
    '''
    pass

def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    shape = X_trans.shape
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn
    
    

def camera_calibration():
    '''
    完整清晰的解释：
    https://github.com/wingedrasengan927/Image-formation-and-camera-calibration
    https://towardsdatascience.com/camera-intrinsic-matrix-with-example-in-python-d79bf2478c12
    其他实验可参考 https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
    P_c = E * P_w
    E :  extrinsic camera matrix
    R: rotation matrix of shape (3, 3)
    O: translation offset of shape (3, 1)
    E = 
    [[R, O], 
    [O^T, 1]]
    
    P` = k * P_c
    P` - Homogeneous coordinates of the point in the image
    k  - Camera Intrinsic Matrix
    P_c - Homogeneous Coordinates of the point in the world wrt camera
    
    k: camera intrinsic matrix
    k = 
    [[f, s, cx],
    [0, af, cy],
    0, 0, 1]]
    
    f      - focal length
    s      - skew factor
    cx, cy   - offset
    a      - aspect ratio
    本论文,s=0,f二分法估计,a=1, cx,cy设置为图形中心,即估计出f,即可.
    P`=k*(E*P_w)
    '''
    pass

def get_shoulder_pts():
    '''
    https://github.com/YuanxunLu/LiveSpeechPortraits/issues/9 关键点检测导致的帧间抖动,也可参考
    2D shoulder points using LK flow and reconstruct 3D shoulder points.
    Specifically, shoulder points are not automatically detected. We manually selected once for the first frame and
    tracked them using optical flow. Learning-based methods, e.g., raft may work better.
    
    The shoulder is modeled as a billboard, the depth of the billboard is set as the average depth of the facial landmarks 
    in the training sequences. Please check section 3.3 in the paper. Yep, tracked shoulder points are 2D at first, but we 
    can reconstruct them into 3D space (using billboard assumption & camera calibration).
    问题:什么是billboard, 参考https://zhuanlan.zhihu.com/p/147342778, 简单来说billboard就是固定朝向摄影机的贴图矩形
    https://www.landontownsend.com/single-post/2019/04/15/shader-experimentation-cut-out-particles-with-depth
    camera calibration 参考: https://towardsdatascience.com/camera-calibration-with-example-in-python-5147e945cdeb,
    https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    首先可读相机标定:https://blog.csdn.net/lql0716/article/details/71973318
    ''' 
    pass


def get_track3d_ptx_fix_contours():
    '''
    The only difference is that I fix the face contour points in the latter file. Contour here means
    that I fix the contour indices for the reconstruction results. This is about another area about 3D 
    face model tracking. In a word, 
    I fixed the contour indices instead of using sliding contour indices found during the tracking
    固定边界下标,参考3D人脸模型追踪,而不是追踪过程中滑动边界indices
    同样,如何转到2d的,transformed.
    '''
    pass

# ffmpeg -i .xx.mp4 -f wav -ar 16000 xx.wav
# ffmpeg -i <input> -filter:v fps=60 <output>  'ffmpeg_%0d.png'
# ffmpeg -i input.mp4 '%04d.png'
# https://blog.csdn.net/bby1987/article/details/108923361
# def write_video_with_audio(audio_path, output_path, prefix='pred_'):
#     fps, fourcc = 60, cv2.VideoWriter_fourcc(*'DIVX')
#     video_tmp_path = join(save_root, 'tmp.avi')
#     out = cv2.VideoWriter(video_tmp_path, fourcc, fps, (Renderopt.loadSize, Renderopt.loadSize))
#     for j in tqdm(range(nframe), position=0, desc='writing video'):
#         img = cv2.imread(join(save_root, prefix + str(j+1) + '.jpg'))
#         out.write(img)
#     out.release() 
#     cmd = 'ffmpeg -i "' + video_tmp_path + '" -i "' + audio_path + '" -codec copy -shortest "' + output_path + '"'
#     subprocess.call(cmd, shell=True) 
#     os.remove(video_tmp_path)  # remove the template video


def extract_video_audio(video_path, save_audio_dir='', sr=16000, fps=60, size=(512, 512)):
    print(video_path, 'xxx')
    name = os.path.basename(video_path).split('.')[0]
    cmd = f'ffmpeg -i {video_path} -f wav -ar {sr} {os.path.join(save_audio_dir, name)}.wav'
    print(cmd)
    subprocess.call([cmd], shell=True)
    cmdv = f'ffmpeg -i {video_path} -filter:v fps={fps} {save_audio_dir}/{name}_%0d.png'
    subprocess.call([cmdv], shell=True)
    
    print("Successed")
    
    
def draw_bbox(img, bbox):
    pass

def crop_face(img, bbox, dst_size=(512, 512)):
    pass

def img2tensor(image):
    '''
    PIL :transform = transforms.Compose([
    transforms.PILToTensor()
    ])
  
    '''
    from torchvision import transforms
    # # Convert BGR image to RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
    transforms.ToTensor()
    ])
    # Convert the image to Torch tensor
    tensor = transform(image)
    return tensor

from decalib.utils.tensor_cropper import Cropper

def crop_video(vd, crop_size=(512, 512), fps=60):
    '''
    将视频以人脸为中心，crop为512,512大小，并以60fps进行保存，声音保持同步
    '''
    # from decalib.datasets.detectors import FAN
    # face_detect = FAN()
    import face_alignment
    
    model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    crop_obj =  Cropper(crop_size[0], scale=[3,2], trans_scale = 0.)
    
    cap = cv2.VideoCapture(vd)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_write = cv2.VideoWriter('g_output_180.mp4', fourcc, fps, crop_size, True)
    # bbox, _ = face_detect.run()  # [left,top, right, bottom]
    # if len(bbox) == 4:
    #     bc = ((bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2)
    # else:
    #     print('no face')
    i = 0
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        out = model.get_landmarks(frame)
        frame = img2tensor(frame)  # np.array to tensor
        points = torch.from_numpy(out[0]).unsqueeze(0)
        cropped_image, tform = crop_obj.crop(frame.unsqueeze(0), points)
        trans_points = crop_obj.transform_points(points, tform, points_scale=None, normalize = True)
        frame = cropped_image.detach().cpu().squeeze().numpy().transpose(1, 2, 0) # tensor to cv2 array , permute(2, 0, 1)
        frame = np.uint8(frame * 255)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('xx_32.png', frame)
        
        out_write.write(frame)
        # if i == 180:
        #     break
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break
        # print('only on frame!')
        # break
    # 完成工作后释放所有内容
    cap.release()
    out_write.release()
    cv2.destroyAllWindows()

def visual():
    pass

def parser_json(jp):
    kp = []
    with open(jp, 'r') as f:
        data = json.load(f)
    shapes = data['shapes']
    for p in shapes:
        ps = p['points'][0]
        kp.append(ps)

    return kp


def lucas_kanade_method_imgs(resources_path,  kp=None, save_kp_dir='./'):

    if kp is None:
        raise ValueError(f'{kp} should not be None, using labelme to get the keypoints what u need on first frame')

    save_kps = []
    if not os.path.exists(save_kp_dir):
        os.makedirs(save_kp_dir, exist_ok=True)
    
    s = lambda x: float(x.split('_')[-1][:-4])  # 按时间帧排序

    imgs = sorted([img for img in os.listdir(resources_path) if img.endswith(".{}".format('png'))], key=s)
    
    frame_first = cv2.imread(os.path.join(resources_path, imgs[0]))

    # Take first frame and find corners in it
    old_gray = cv2.cvtColor(frame_first, cv2.COLOR_BGR2GRAY)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # append the first kp
    kp_array = []
    for p in kp:
        kp_array.append(np.array(p).reshape(1, 2))
    p0 = np.array(kp_array, dtype=np.float32)
    save_kps.append(p0.reshape(-1, 2))

    mask = np.zeros_like(frame_first)
    # Create a mask image for drawing purposes
    color = [0, 0, 255]
    img_nums = len(imgs)

    save_randome = np.random.choice(range(img_nums), 10, replace=False)
    print(save_randome)
    # save_randome = random.sample(range(img_nums), 10)

    for i, img in enumerate(imgs[1:]):
        frame_after = cv2.imread(os.path.join(resources_path, img))
        # frame = frame_after.copy()

        frame_gray = cv2.cvtColor(frame_after, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if i in save_randome:
            # Draw the tracks
            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = list(map(int, [a, b, c, d]))
                mask = cv2.line(mask, (a, b), (c, d), color, 2)
                frame = cv2.circle(frame_after, (a, b), 5, color, -1)

            # Display the demo
            img_draw = cv2.add(frame, mask)
            cv2.imwrite(os.path.join(save_kp_dir, f'save_random_kp_{i}.png'), img_draw)
            # cv2.imshow("frame", img)
            # k = cv2.waitKey(25) & 0xFF
            # if k == 27:
            #     break

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        save_kps.append(good_new)
        p0 = good_new.reshape(-1, 1, 2)
    save_kps = np.array(save_kps, dtype="object")
    print(len(save_kps))

    npy_save_path =os.path.join(save_kp_dir, 'shoulder_2D.npy')
    # if not os.path.exists(npy_save_path):
    np.save(npy_save_path, save_kps)

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1 
def plot_kpts(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()
    radius = max(int(min(image.shape[0], image.shape[1])/200), 1)
    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        if kpts.shape[1]==4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        if i in end_list:
            continue
        ed = kpts[i + 1, :2]
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), (255, 255, 255), radius)
        image = cv2.circle(image,(int(st[0]), int(st[1])), radius, c, radius*2)
        # image = cv2.putText(image, f'{i}', org=(int(st[0]), int(st[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2, color=c, thickness=1) 

    return image


def rot_angle(rot, angle):
    pass


def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, crop_size=224, face_detector=args.detector)

    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config = deca_cfg, device=device)
    cam_flag = True
    original_image = testdata[0]['original_image'].cpu().numpy()
    _, ho, wo = original_image.shape  # torch-> cv2 or skimage in channel
    
    pts_3d = []  # (x, 68, 3)
    pts_3d_224 = []
    rot_angles = []  # (x, 3), headpose=[rot_angles, trans]
    trans = []  # (x, 3, 1), 
    three_fit_data = []
    # get_3d_fit_data()
    
    # f, cx, cy
    # get_camera_intrinsic()
    
    tracked_pts_2d = [] 
    tracked_pts_3d = [] 
    tforms = []
    landmarks2d_original = []
    cams = []
    
    # camera_calibration
    # https://stackoverflow.com/questions/19679703/projecting-a-2d-point-into-3d-space-using-camera-calibration-parameters-in-openc
    # https://stackoverflow.com/questions/55734940/how-to-perform-2d-to-3d-reconstruction-considering-camera-calibration
    # (x, 18, 2) -> (x, 18, 3)  目前还不知道如何解，从facedata看来，3D没被使用
    # get_shoulder_pts()
    
    # (x, 68, 2, (x, 68, 3)
    # get_track3d_ptx_fix_contours()
    
    # scale, xc, yc
    # get_change_paras()

    
    for i in tqdm(range(len(testdata))):
        images = testdata[i]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)
            # print(codedict.keys()) # dict_keys(['shape', 'tex', 'exp', 'pose', 'cam', 'light', 'images', 'detail'])
            cam = codedict['cam']
            cams.append(cam.cpu().numpy())
            # print(cam, 'cccam')  # every frame has diff cam
            pose = codedict['pose']
            # print(cam, pose, pose.shape, cam.shape, images.shape, light.shape)
            opdict = deca.decode_simple(codedict)
            # opdict, visdict = deca.decode(codedict, render_orig=False)
            
            rot_angles.append(pose[0,:3])
            trans.append(pose[0,3:])
            
            tform = testdata[i]['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1,2).to(device)
            tforms.append(tform.cpu().numpy())
            
            # transform camera to oringal coordinate 
            if cam_flag:
                cam = codedict['cam'].detach().cpu().numpy()  # [scale, tx, ty], in [-1, 1] ? or [0,1] ?
                save_cam = {}
                save_cam["scale"], save_cam["xc"], save_cam["yc"] = cam[0,0], cam[0,1], cam[0,2]
                np.save(os.path.join(savefolder, 'camera_extrinsics.npy'), save_cam)
                cam_flag = False
            
            # add landmark2d on rriginal image
            landmarks2d = opdict['landmarks2d']  # 1, 68, 3]
            landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]
            landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]  # y -1? 
            landmarks2d = transform_points(landmarks2d, tform, points_scale=[224, 224], out_scale=[h, w])
            
            landmarks3d = opdict['landmarks3d']
            # landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam'])
            # landmarks3d[:,:,1:] = -landmarks3d[:,:,1:]  # y, z -1 ?
            landmarks3d = util.batch_orth_proj_(landmarks3d, codedict['cam'], inverse=False)
            
            pts_3d_224.append(landmarks3d)
            
            landmarks3d = transform_points(landmarks3d, tform, points_scale=[224, 224], out_scale=[h, w])
            
            pts_3d.append(landmarks3d)
            
            # normalized_pose = util.batch_orth_proj(pose[0, :3], codedict['cam'])
            # print(normalized_pose)
            # rot_angles.append(normalized_pose)

        # if  args.saveKpt or args.saveImages:
        #     os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        
        landmarks2d_original.append(landmarks2d.cpu().numpy())
        
        if i == 200:
            break
        
        # landmarks2d_original.append(landmarks2d)
        # landmarks3d = transform_points(opdict['landmarks3d'], tform, points_scale=[224, 224], out_scale=[h, w])
        # landmarks2d = landmarks3d[..., :2]  # 等价
    
        if args.saveKpt:
            # rot_angles, trans test 

            # landmarks2d[..., 0] = landmarks2d[..., 0] * wo/2 + wo/2
            # landmarks2d[..., 1] = landmarks2d[..., 1] * ho/2 + ho/2
            landmarks3d[..., 0] = landmarks3d[..., 0] * wo/2 + wo/2
            landmarks3d[..., 1] = landmarks3d[..., 1] * ho/2 + ho/2
  
            original_image = testdata[i]['original_image'].cpu().numpy()
            img = np.uint8(original_image*255)  # CHW, RGB
            '''
            # rgb = bgr[...,::-1]
            # bgr = rgb[...,::-1]
            # gbr = rgb[...,[2,0,1]]
            # chw->hwc, hwc->chw: transpose(2, 0, 1)
            # img = cv2.imread("001.jpg") # BGR
            # img_ = img[:,:,::-1].transpose((2,0,1))  # BGR->RGB->CHW
            '''
            img = img.transpose(1, 2, 0)[..., ::-1]  # CHW->HWC->BGR
            # choosed_landmark = landmarks2d[0, ...].cpu().numpy()
            choosed_landmark = landmarks3d[0, ...].cpu().numpy()
            
            # img = util.plot_kpts(img, choosed_landmark)  # add line
            if i <= 200:
                img = plot_kpts(img, choosed_landmark)
                
                # cv2.imwrite('kh_new_BGR_tformed_has_orth_proj_2d.png', img)  # right
                cv2.imwrite(f'./out/kh_orth_proj_3d_rot_trans_2d_{i}.png', img)
                
            
        # if i == 2:
        #     break
        
    # img_meta = {'h':ho, 'w':wo}
    # # # # print(img_meta)
    # np.save(os.path.join(savefolder, 'img_meta.npy'), img_meta)
    # f = lambda x: torch.stack(x, dim=0).cpu().numpy()
    # pts_3d, rot_angles, trans = f(pts_3d), f(rot_angles), np.expand_dims(f(trans), axis=-1)
    # pts_3d_224 = f(pts_3d_224)
    # mean_pts_3d = np.mean(pts_3d, axis=0)
    # # print(mean_pts_3d.shape, pts_3d.shape)
    # np.save(os.path.join(savefolder, 'mean_pts_3d.npy'), mean_pts_3d)
    # # print(pts_3d.shape, rot_angles.shape, trans.shape)
    # np.savez(os.path.join(savefolder, '3d_fit_data.npz'), pts_3d=pts_3d, rot_angles=rot_angles, trans=trans)  # TODO 
    
    # np.save(os.path.join(savefolder, 'landmarks2d_original.npy'), landmarks2d_original)
    # np.save(os.path.join(savefolder, 'tforms.npy'), tforms)
    # np.save(os.path.join(savefolder, 'cams.npy'), cams)
    
    # np.savez(os.path.join(savefolder, '3d_fit_data_224.npz'), pts_3d=pts_3d_224, rot_angles=rot_angles, trans=trans) 
    
    # print(len(landmarks2d_original), len(tforms), cams)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation for generat lsp data')
    parser.add_argument('--video_path', type=str, default='', help='')
    parser.add_argument('--video_img_save', type=str, default='' , help='')
    parser.add_argument('--save_audio_dir', type=str, default='./' , help='')
    parser.add_argument('-i', '--inputpath', default='./kanghui_5_imgs')
    parser.add_argument('-s', '--savefolder',default='')
    
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works \
                            when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    
    
    args = parser.parse_args()
    
    # if not os.path.exists(args.video_img_save):
    #     os.makedirs(args.video_img_save, exist_ok=True)
        
    # get wav and 60 fps images
    # (720, 1280, 3)
    # extract_video_audio(args.video_path, args.video_img_save) 
    # print(args.video_path, args.video_img_save)
    # get cam, 
    
    # # get shoulder points
    # kp = parser_json('./demos/kanghui_5_60fps_crop_512_1.json')
    # lucas_kanade_method_imgs(args.video_img_save, kp, save_kp_dir='./kanghui_512_LSP')
    
    main(args)
    
    # crop_video('../g_60fps.mp4')