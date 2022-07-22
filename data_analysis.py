

from lib.utils.visisual import plot_kpts, frames2video
import numpy as np
import cv2
import os


def ndc2img(landmarks3d):
    
    landmarks3d[...,0] = landmarks3d[...,0]*256 + 256
    landmarks3d[...,1] = landmarks3d[...,1]*256 + 256
    
    return landmarks3d[..., :2]


def iter_frames(path, add_mean=False):
    landmarks = np.load(path)
    pts_3d, rot_angles, trans = landmarks['pts_3d'], landmarks['rot_angles'], landmarks['trans']
    #  add mean
    if add_mean:
        path_mean = './data/hk_fake_38/label/mean_pts_3d.npy'
        mean_landmarks = np.load(path_mean)
        landmark_mean = ndc2img(mean_landmarks)
    img = np.zeros((512, 512, 3),  dtype=np.uint8)
    for pt3d in pts_3d:
        points = ndc2img(pt3d[0])
        img_landmark = plot_kpts(img, points)
        if add_mean:
            img_landmark = plot_kpts(img_landmark, landmark_mean[0])
        yield img_landmark
        

def get_landmarks_video(path):
    frames = iter_frames(path, add_mean=True)
    frames2video('./original_add_mean_landmarks.mp4', frames)


def draw_face_feature_maps(keypoints, size=(512, 512)):
    # 73 landmarks
    part_list = [[list(range(0, 15))],                                # contour
                    [[15,16,17,18,18,19,20,15]],                         # right eyebrow
                    [[21,22,23,24,24,25,26,21]],                         # left eyebrow
                    [range(35, 44)],                                     # nose
                    [[27,65,28,68,29], [29,67,30,66,27]],                # right eye
                    [[33,69,32,72,31], [31,71,34,70,33]],                # left eye
                    [range(46, 53), [52,53,54,55,56,57,46]],             # mouth
                    [[46,63,62,61,52], [52,60,59,58,46]]                 # tongue
                ]
    w, h = size
    # edge map for face region from keypoints
    im_edges = np.zeros((h, w, 3), np.uint8) # edge map for all edges
    for edge_list in part_list:
        for edge in edge_list:
            for i in range(len(edge)-1):
                st, ed = edge[i], edge[i+1]
                p1 = (int(keypoints[st, :][0]), int(keypoints[st, :][1]))
                p2 = (int(keypoints[ed, :][0]), int(keypoints[ed, :][1]))
                im_edges = cv2.line(im_edges, p1, p2, (255, 0, 0), 2)

    return im_edges

def iter_tracked_npy(kps):
    for kp in kps:
        im_edges = draw_face_feature_maps(kp)
        yield im_edges


def visual_orignal(base_dir='./data/Obama2'):
    pts_2d = np.load(os.path.join(base_dir, 'tracked2D_normalized_pts_fix_contour.npy'))
    pts_3d = np.load(os.path.join(base_dir, 'tracked3D_normalized_pts_fix_contour.npy'))
    pts_3d = pts_3d[..., :2] * 256 + 256
    pts_3d[..., 1] = 512 - pts_3d[..., 1] 
    pts = pts_3d
    
    # kp = pts_2d[0]
    # im_edges = draw_face_feature_maps(kp)
    # cv2.imwrite('./xx_gg.png', im_edges)
    
    frames = iter_tracked_npy(pts)
    frames2video('./lsp_track_landmarks_3d.mp4', frames)
    
from math import cos, sin
def angle2matrix(angles, gradient='false'):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
        gradient(str): whether to compute gradient matrix: dR/d_x,y,z
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x), -sin(x)],
                 [0, sin(x),  cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    
    R=Rz.dot(Ry.dot(Rx))
    #R=Rx.dot(Ry.dot(Rz))
    
    if gradient != 'true':
        return R.astype(np.float32)

    
def fix_contur(locate=True):
    
    counter_index = range(17)
    path = './data/hk_fake_38/label/3d_fit_data_224.npz'
    data_info = np.load(path)
    pts_3d, trans, rots = data_info['pts_3d'], data_info['trans'], data_info['rot_angles']  
    pts_3d = np.squeeze(pts_3d)
    # print(pts_3d.shape,  trans.shape, rots.shape)  # (18415, 1, 68, 3) (18415, 3, 1) (18415, 3)
    new_pts_3d = np.zeros_like(pts_3d)
    # rots_rad2deg = rots / np.pi * 180.
    # first_counter = pts_3d[0, :, counter_index, :]
    
    # print(pts_3d[0], rots[0], rots_rad2deg[0], trans[0])
    # 这里有多种可能， 世界坐标的平移和旋转， 局部坐标的平移的旋转
    img = np.zeros((512, 512, 3),  dtype=np.uint8)
    for i in range(pts_3d.shape[0]):
        R = angle2matrix(rots[i])
        if locate:
            translate_rotation = R.T.dot(pts_3d[i].T) - trans[i]  # R.dot(pts_3d[i].T) + trans[i]  # 不排除T,R使用错误。
            translate_rotation = translate_rotation.T
        else:
            pass
        points = ndc2img(translate_rotation)
        img_edge = plot_kpts(img, points)
        yield img_edge
    

def test_pts_3d_rot_trans():
    frames = fix_contur()
    frames2video('orignal_pts3d_trans_rotation.mp4', frames)

    
if __name__ == "__main__":
    path = './data/hk_fake_38/label/3d_fit_data.npz'  # landmarks2d_original.npy, mean_pts_3d.npy, 3d_fit_data.npz
    path_mean = './data/hk_fake_38/label/mean_pts_3d.npy'
    
    # get_landmarks_video(path)
    
    # visual_orignal()
    # fix_contur()
    test_pts_3d_rot_trans()
    
    # how to fix the contour 

