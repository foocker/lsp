from lib.utils.visisual import (
    plot_kpts,
    frames2video,
    draw_face_feature_maps as dffm_68,
)
from lib.datasets.face_utils import mounth_open2close
from superpose3d import Superpose3D
import numpy as np
import math
import cv2
import os
from os.path import join as pjoin
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline

from math import cos, sin, atan2


def ndc2img(landmarks3d):

    landmarks3d[..., 0] = landmarks3d[..., 0] * 256 + 256
    landmarks3d[..., 1] = landmarks3d[..., 1] * 256 + 256

    return landmarks3d[..., :2]


def iter_frames(path, add_mean=False):
    landmarks = np.load(path)
    pts_3d, rot_angles, trans = (
        landmarks["pts_3d"],
        landmarks["rot_angles"],
        landmarks["trans"],
    )
    #  add mean
    if add_mean:
        path_mean = "./data/hk_fake_38/label/mean_pts_3d.npy"
        mean_landmarks = np.load(path_mean)
        landmark_mean = ndc2img(mean_landmarks)
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for pt3d in pts_3d:
        points = ndc2img(pt3d[0])
        img_landmark = plot_kpts(img, points)
        if add_mean:
            img_landmark = plot_kpts(img_landmark, landmark_mean[0])
        yield img_landmark


def get_landmarks_video(path):
    frames = iter_frames(path, add_mean=True)
    frames2video("./original_add_mean_landmarks.mp4", frames)


def draw_face_feature_maps(keypoints, size=(512, 512)):
    # 73 landmarks
    part_list = [
        [list(range(0, 15))],  # contour
        [[15, 16, 17, 18, 18, 19, 20, 15]],  # right eyebrow
        [[21, 22, 23, 24, 24, 25, 26, 21]],  # left eyebrow
        [range(35, 44)],  # nose
        [[27, 65, 28, 68, 29], [29, 67, 30, 66, 27]],  # right eye
        [[33, 69, 32, 72, 31], [31, 71, 34, 70, 33]],  # left eye
        [range(46, 53), [52, 53, 54, 55, 56, 57, 46]],  # mouth
        [[46, 63, 62, 61, 52], [52, 60, 59, 58, 46]],  # tongue
    ]
    w, h = size
    # edge map for face region from keypoints
    im_edges = np.zeros((h, w, 3), np.uint8)  # edge map for all edges
    for edge_list in part_list:
        for edge in edge_list:
            for i in range(len(edge) - 1):
                st, ed = edge[i], edge[i + 1]
                p1 = (int(keypoints[st, :][0]), int(keypoints[st, :][1]))
                p2 = (int(keypoints[ed, :][0]), int(keypoints[ed, :][1]))
                im_edges = cv2.line(im_edges, p1, p2, (255, 0, 0), 2)

    return im_edges


def iter_tracked_npy(kps):
    for kp in kps:
        im_edges = draw_face_feature_maps(kp)
        yield im_edges


def visual_orignal(base_dir="./data/Obama2"):
    pts_2d = np.load(os.path.join(base_dir, "tracked2D_normalized_pts_fix_contour.npy"))
    pts_3d = np.load(os.path.join(base_dir, "tracked3D_normalized_pts_fix_contour.npy"))
    pts_3d = pts_3d[..., :2] * 256 + 256
    pts_3d[..., 1] = 512 - pts_3d[..., 1]
    pts = pts_3d

    # kp = pts_2d[0]
    # im_edges = draw_face_feature_maps(kp)
    # cv2.imwrite('./xx_gg.png', im_edges)

    frames = iter_tracked_npy(pts)
    frames2video("./lsp_track_landmarks_3d.mp4", frames)



def angle2matrix(angles):
    """get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left.
        z: roll. positive for tilting head right.
    Returns:
        R: [3, 3]. rotation matrix.
    """
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))

    return R.astype(np.float32)


def angle2matrix_torch(angles):
    import torch

    """ get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [batch_size, 3] tensor containing X, Y, and Z angles.
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [batch_size, 3, 3]. rotation matrices.
    """
    angles = angles * (np.pi) / 180.0
    s = torch.sin(angles)
    c = torch.cos(angles)

    cx, cy, cz = (c[:, 0], c[:, 1], c[:, 2])
    sx, sy, sz = (s[:, 0], s[:, 1], s[:, 2])

    zeros = torch.zeros_like(s[:, 0]).to(angles.device)
    ones = torch.ones_like(s[:, 0]).to(angles.device)

    # Rz.dot(Ry.dot(Rx))
    R_flattened = torch.stack(
        [
            cz * cy,
            cz * sy * sx - sz * cx,
            cz * sy * cx + sz * sx,
            sz * cy,
            sz * sy * sx + cz * cx,
            sz * sy * cx - cz * sx,
            -sy,
            cy * sx,
            cy * cx,
        ],
        dim=0,
    )  # [batch_size, 9]
    R = torch.reshape(R_flattened, (-1, 3, 3))  # [batch_size, 3, 3]
    return R


def fix_contur(locate=True):

    counter_index = range(17)
    path = "./data/hk_fake_38/label/3d_fit_data_224.npz"
    data_info = np.load(path)
    pts_3d, trans, rots = (
        data_info["pts_3d"],
        data_info["trans"],
        data_info["rot_angles"],
    )
    pts_3d = np.squeeze(pts_3d)
    # print(pts_3d.shape,  trans.shape, rots.shape)  # (18415, 1, 68, 3) (18415, 3, 1) (18415, 3)
    new_pts_3d = np.zeros_like(pts_3d)
    # rots_rad2deg = rots / np.pi * 180.
    # first_counter = pts_3d[0, :, counter_index, :]

    # print(pts_3d[0], rots[0], rots_rad2deg[0], trans[0])
    # 这里有多种可能， 世界坐标的平移和旋转， 局部坐标的平移的旋转
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(pts_3d.shape[0]):
        R = angle2matrix(rots[i])
        if locate:
            translate_rotation = (
                R.T.dot(pts_3d[i].T) - trans[i]
            )  # R.dot(pts_3d[i].T) + trans[i]  # 不排除T,R使用错误。
            translate_rotation = translate_rotation.T
        else:
            pass
        # TODO here should add camera intric? 
        points = ndc2img(translate_rotation)
        img_edge = plot_kpts(img, points)
        yield img_edge


def test_pts_3d_rot_trans():
    frames = fix_contur()
    frames2video("orignal_pts3d_trans_rotation.mp4", frames)


def get_optim_points(mode="o"):
    """
    选择固定度比较强的点,使得性能最好,前17个表示边缘, 当前增加了鼻子线
    """
    path = "./data/hk_fake_8_14/label/3d_fit_data.npz"
    data_info = np.load(path)
    pts_3d = data_info["pts_3d"]
    pts_3d = np.squeeze(pts_3d)
    index = list(range(17)) + list(range(27, 36))
    return pts_3d[0, index, :], pts_3d[:, index, :], pts_3d


def inversetransform(R, T, S, src, dst):
    """
    保持边缘不变,嘴部和眼睛运动保持的映射, 替换. RTS is countour(src)->fixed countour(dst),
    here is src->dst is all points transform to dst, but keep other moving in the src for
    the rest points.
    3*3, 3*1. 1. 68*3, 68*3
    src: mobile_landmarks_all, dst: frozen_landmarks_all
    """
    _x = src.transpose()
    _xprime = S * np.matmul(R, _x) + np.outer(T, np.array([1] * len(dst)))
    # _xprime = _x +  np.outer(T, np.array([1] * len(dst)))
    # _xprime = np.matmul(R.T, _xprime)
    xprime = _xprime.transpose()

    return xprime


def get_RTS(frozen_landmarks, mobile_landmarks, W=None, allow_rescale=False):
    result = Superpose3D(
        frozen_landmarks,
        mobile_landmarks,
        W,
        allow_rescale=allow_rescale,
        report_quaternion=False,
    )
    R = result[1]
    T = result[2].transpose()
    S = result[3]
    return R, T, S


def get_fix_landmarks():
    fix_countour_pts_3d_path = (
        "./data/hk_fake_8_14/label/landmarks3d_fixed_countour_nose.npy"
    )
    fix_mean = "./data/hk_fake_8_14/label/mean_pts_3d_fixed_countour_nose.npy"
    fix_mean_close_mouth = (
        "./data/hk_fake_8_14/label/mean_pts_3d_fixed_countour_nose_closemouth.npy"
    )
    fix_countour_pts_3d_data = []
    frozen_landmarks, mobile_landmarks, pts_3d = get_optim_points()
    for i in range(len(pts_3d)):
        R, T, S = get_RTS(
            frozen_landmarks, mobile_landmarks[i, ...], allow_rescale=True
        )
        xprime = inversetransform(R, T, S, pts_3d[i, ...], pts_3d[0, ...])
        fix_countour_pts_3d_data.append(xprime)

    np.save(fix_countour_pts_3d_path, fix_countour_pts_3d_data)
    mean_pts = np.array(fix_countour_pts_3d_data).mean(axis=0)
    mean_pts_close = mounth_open2close(mean_pts)
    np.save(fix_mean, mean_pts)
    np.save(fix_mean_close_mouth, mean_pts_close)


def iter_fix_countour():
    # landmarks3d_fixed = np.load('./data/hk_fake_38/label/landmarks3d_fixed_countour_nose.npy')
    # fixed_path = pjoin(base_dir, 'label', file_name) # base_dir, file_name
    # landmarks3d_fixed = np.load('./data/hk_fake_8_14/label/landmarks3d_fixed_countour_nose.npy')
    landmarks3d_fixed = np.load(
        "./data/hk_fake_8_14/label/pose_mat_fixed_landmarks.npy"
    )
    print(len(landmarks3d_fixed))
    p_mean = np.load(
        "./data/hk_fake_8_14/label/mean_pts_3d_fixed_countour_nose_closemouth.npy",
        allow_pickle=True,
    )
    p_mean = ndc2img(p_mean)
    p_mean[:, 1] += 45
    for p in landmarks3d_fixed[:4640]:
        points = ndc2img(p)
        img_edge = dffm_68(points)
        img_edge_mean = dffm_68(p_mean)
        # img_edge = (img_edge + img_edge_mean)//2
        yield img_edge


def test_fix_countour():
    frames = iter_fix_countour()
    # frames2video('landmarks3d_fixed_countour_nose_zero.mp4', frames)
    # frames2video('fixed_countour_nose_add_mean_close_mouth.mp4', frames)
    frames2video("pose_mat_fixed_landmarks.mp4", frames)


def test_superpose3d_facelandmarks(
    frozen_landmarks, mobile_landmarks, W=None, allow_rescale=False
):
    result = Superpose3D(
        frozen_landmarks,
        mobile_landmarks,
        W,
        allow_rescale=allow_rescale,
        report_quaternion=False,
    )
    R = np.array(result[1])
    T = np.array(result[2]).transpose()
    c = result[3]
    rmsd = result[0]

    _x = np.array(mobile_landmarks).transpose()
    _xprime = c * np.matmul(R, _x) + np.outer(T, np.array([1] * len(frozen_landmarks)))
    xprime = np.array(_xprime.transpose())

    RMSD = 0.0
    sum_w = 0.0
    if W is None:
        W = [1 for _ in range(len(frozen_landmarks))]
    for i in range(0, len(frozen_landmarks)):
        RMSD += W[i] * (
            (frozen_landmarks[i][0] - xprime[i][0]) ** 2
            + (frozen_landmarks[i][1] - xprime[i][1]) ** 2
            + (frozen_landmarks[i][2] - xprime[i][2]) ** 2
        )
        sum_w += W[i]

    diff = xprime - frozen_landmarks
    RMSD = np.sqrt(RMSD / sum_w)

    # print('solover rmsd is ', rmsd, 'infer rmsd is', RMSD)
    # print('landmark diff is: ', diff)
    if RMSD > 1e-2:
        print(RMSD)
        return True
    return


class HeadPose(object):
    def __init__(self, landmarks2d, landmarks3d, **kwargs) -> None:
        super().__init__()
        """
        landmarks3d.shape = [n, 68, 3]
        68 landmarks, 3d and its correspond 2d points, and the camera matrix, can pnp to get R, T 
        
        s[x, y, z]^T = [[f_x, 0, c_x],
                        [0 f_y, c_y],
                        [0, 0, 1] ] * [R | T] * [U, V, W, 1]^T
        method can replace by: FSA-Net
        
        1. 旋转矩阵转欧拉角，反之
        2. 罗德里格斯旋转向量转旋转矩阵，反之。
        3. 具体旋转案例,齐次坐标
        """
        self.landmarks3d = landmarks3d
        self.landmarks2d = landmarks2d
        assert self.landmarks2d.shape[1] == self.landmarks3d.shape[1] == 68
        self.six_points_index = [8, 30, 36, 45, 48, 54]  # 下巴, 鼻尖, 左眼角, 右眼角, 左嘴角, 右嘴角
        self.fourteen_points_index = [
            17,
            21,
            22,
            26,
            36,
            39,
            42,
            45,
            31,
            35,
            48,
            54,
            57,
            8,
        ]
        # for k, v in kwargs.items():
        #     if
        img_shape = kwargs.get("image_shape")
        assert img_shape is not None, "img_shape is None"
        h, w = img_shape[0], img_shape[1]
        focal_lenght = h
        center = (h // 2, w // 2)
        self.camera_instric_matrix = np.array(
            [[focal_lenght, 0, center[0]], [0, focal_lenght, center[1]], [0, 0, 1]],
            dtype=np.float32,
        )
        self.dist_coeffs = np.zeros((4, 1))

    def get_head_pose(self, landmark2d, landmark3d, point_num=14):
        """
        handle one landmark
        """
        if point_num == 14:
            img_pts = landmark2d[self.fourteen_points_index]
            obj_pts = landmark3d[self.fourteen_points_index]  # np.float32

        elif point_num == 6:
            img_pts = landmark2d[self.six_points_index]
            obj_pts = landmark3d[self.six_points_index]

        else:
            img_pts = landmark2d
            obj_pts = landmark3d
        obj_pts = obj_pts.astype("float32")
        img_pts = img_pts.astype("float32")

        # cv2.solvePnPRansac()
        _, rotation_vec, translation_vec = cv2.solvePnP(
            obj_pts, img_pts, self.camera_instric_matrix, self.dist_coeffs
        )

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 旋转向量转为旋转矩阵
        # rmat = angle2matrix(rotation_vec)   # 旋转向量非角度单位, 正确转换为angle2matrix(euler_angle)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # [R | T] 矩阵
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)  # 欧拉角
        
        # ----- just test the correctness of others function ----- 
        # rmat = self.eulerangle_to_rmatrix(euler_angle)  # right: inverse, and orthogonal
        # xyz_angle = self.rmatrix_to_eulerangle(rmat)   # right
        # print(an - euler_angle, 'hk')
        # if self.is_orthogonal_matirx(rotation_mat):  # rmat, rotation_mat, xyz_angle
        #     print('xxx')
        
        # _, _, vec = self.rmatrix_to_rodrigues_rotation_vec(rotation_mat)  # right
        # print(rotation_vec - vec, vec.shape, rotation_vec.shape)
        
        # _, _, vec = self.rmatrix_to_rodrigues_rotation_vec(rmat)  # right
        # print(rotation_vec - vec, vec.shape, rotation_vec.shape)
        
        # Rx = self.rodrigues_rotation_vec_to_rmatrix(rotation_vec)   # right
        # if self.is_orthogonal_matirx(Rx):
        #     print('xxx')

        # angle, _, vec = self.rmatrix_to_rodrigues_rotation_vec(Rx)  # right
        # print(Rx - rmat, vec - rotation_vec)

        # angle2 = self.rodrigues_rotation_vec_to_eulerangle(rotation_vec)  # right
        # print('euler angel diff: ', angle2 - xyz_angle)
        # ----- just test the correctness of others function ----- 
        
        return pose_mat, euler_angle

    def inverse_rt(self, landmark3d, R, T, mode=True):
        """
        将物体坐标关于世界坐标的旋转和平移去除,使得原始物体保持不动的效果.正确性待测试.
        """
        if mode:
            fixed_land3d = np.linalg.inv(R).dot(landmark3d.T) - T.T
        else:
            fixed_land3d = R.dot(landmark3d.T) + T.T
        return fixed_land3d.T

    def test_rt(self, save_p=None, mode=True):
        fixed_landmarks3d = np.zeros_like(
            self.landmarks3d, dtype=self.landmarks3d.dtype
        )
        i = 0
        for land2, land3 in zip(self.landmarks2d, self.landmarks3d):
            pose_mat, _ = self.get_head_pose(land2, land3)
            R = pose_mat[:3, :3]
            T = pose_mat[:3, 3][np.newaxis, :]
            fixed_land3d = self.inverse_rt(land3, R, T, mode=mode)
            fixed_landmarks3d[i] = fixed_land3d
            i += 1
        fixed_landmarks3d = np.array(fixed_landmarks3d, dtype=np.float32)
        if save_p is not None:
            np.save(save_p, fixed_landmarks3d)
            print("Saving Successed.")
        return fixed_landmarks3d

    def save_np(self, npdata, save_path):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, npdata)

    def is_orthogonal_matirx(self, R):
        Idendity = np.dot(R.T, R)
        I = np.identity(3, dtype=R.dtype)
        diff = np.linalg.norm(I - Idendity)
        return diff < 1e-6

    def rmatrix_to_eulerangle(self, R):
        assert self.is_orthogonal_matirx(R)

        # sz = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        sy = math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)
        singular = sy < 1e-6  # \beta is -+ 90^{\circle} or not
        if not singular:
            # np.arctan2
            x = atan2(R[2, 1], R[2, 2])
            y = atan2(-R[2, 0], sy)
            z = atan2(R[1, 0], R[0, 0])
        else:
            # \beta is -90
            x = atan2(-R[1, 2], R[1, 1])
            y = atan2(-R[2, 0], sy)
            z = 0

        x = x * 180. / math.pi  # np.deg2rad()
        y = y * 180. / math.pi
        z = z * 180. / math.pi

        return np.array([x, y, z]).reshape(3, 1)
    
    def eulerangle_to_rmatrix(self, angle):
        theta = angle * math.pi / 180.0
        theta = theta.reshape(3,)
        x, y, z = theta
        R_x = np.array([[1, 0, 0],
                        [0, cos(x), -sin(x)],
                        [0, sin(x), cos(x)]
                        ])
        R_y = np.array([[cos(y), 0, sin(y)],
                        [0, 1, 0],
                        [-sin(y), 0, cos(y)]
                        ])
        R_z = np.array([[cos(z), -sin(z), 0],
                        [sin(z), cos(z), 0],
                        [0, 0, 1]
                        ])
        R = R_z.dot(R_y.dot(R_x))
        return R.astype(np.float32)
    
    def rodrigues_rotation_vec_to_rmatrix(self, vec):
        # 旋转向量转旋转矩阵
        if vec.shape != (3, 1):
            vec = vec.reshape(3, 1)  # vec[:, None]
        theta = np.linalg.norm(vec)  # 模长为角度，
        r = vec / theta  # 方向向量为旋转轴
        
        # 给定旋转轴和旋转角，返回旋转矩阵
        r = np.array(r).reshape(3, 1)
        rx, ry, rz = r[:, 0]
        M = np.array([[0, -rz, ry],
                      [rz, 0, -rx],
                      [-ry, rx, 0]])
        # R = np.eye(4)
        R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta))*r@r.T + np.sin(theta) * M
        return R
    
    def rmatrix_to_rodrigues_rotation_vec(self, R):
        # reference zhihu/p/401806150
        J = (R - R.T) / 2
        t = [-J[1, 2], J[0, 2], -J[0, 1]]
        t_norm = np.linalg.norm(t)
        theta = np.arcsin(t_norm)
        angle = theta * 180. / np.pi  # 旋转角度
        r = t / t_norm   # 旋转轴
        vec = r * theta  # 旋转向量
        return angle, r, vec.reshape(3, 1)
    
    def rodrigues_rotation_vec_to_eulerangle(self, vec):
        R = self.rodrigues_rotation_vec_to_rmatrix(vec)
        angle = self.rmatrix_to_eulerangle(R)
        return angle
        

    def __getitem__(self, index):
        return self.landmarks2d[index], self.landmarks3d[index]

    def __len__(self):
        return self.landmarks3d.shape[0]


class SmoothScatterPoints(object):
    def __init__(self) -> None:
        super().__init__()
        """
        Savitzky-Golay filter, 滑动平均录滤波,插值法,卡尔曼滤波
        """

    def SG(self, *args, **kwargs):
        result = savgol_filter(*args, **kwargs)
        return result

    def move_avg(self, a, n, mode="full"):
        # full, valid, same
        return np.convolve(a, np.ones((n,)) / n, mode=mode)

    def interp(self, x, y, dense_nnum=10):
        x_smooth = np.linspace(x.min(), x.max(), dense_nnum)
        return make_interp_spline(x, y)(x_smooth)

    def KalmanFilter(self):
        pass


if __name__ == "__main__":
    test_list = {"pose_mat": 1, "fix_land": 2}
    case = test_list.get("pose_mat", None)
    print(case)

    path = "./data/hk_fake_8_14/label/3d_fit_data.npz"  # landmarks2d_original.npy, mean_pts_3d.npy, 3d_fit_data.npz
    path_mean = "./data/hk_fake_8_14/label/mean_pts_3d.npy"

    # get_landmarks_video(path)

    # visual_orignal()
    # fix_contur()
    # test_pts_3d_rot_trans()

    # how to fix the contour
    # get_fix_landmarks()

    # test_fix_countour()

    # mean_pts = np.load(path_mean)
    # points = ndc2img(mean_pts)
    # img_edge = dffm_68(points)
    # cv2.imwrite('./mean_pts_8_14.png', img_edge)

    if case == 1:
        p3 = "./data/hk_fake_8_14/label/3d_fit_data.npz"
        p2 = "./data/hk_fake_8_14/label/landmarks2d_original.npy"
        d3 = np.load(p3)
        l3 = d3["pts_3d"]
        rot = d3["rot_angles"]
        trans = d3["trans"]
        l2 = np.load(p2)
        l2 = np.squeeze(l2)
        l2 = l2 * 256 + 256
        l3 = np.squeeze(l3)

        print(l2.shape, l3.shape)

        HP = HeadPose(l2, l3, image_shape=(512, 512))

        # get pose_mat
        pose_mat_14_data = []
        for i in range(len(HP)):
            land2, land3 = HP[i]
            pose_mat, euler_angle = HP.get_head_pose(
                landmark2d=land2, landmark3d=land3, point_num=14
            )
            pose_mat_14_data.append(pose_mat)
            # R = pose_mat[:3, :3]
            # T = pose_mat[:3, 3]
            # print('rot:', rot[i],'\n', 'trans:', trans[i], '\n','pose_mat:', pose_mat, '\n euler_angle:', euler_angle)
            if i == 10:
                break
        # pose_mat_14_data = np.array(pose_mat_14_data, dtype=np.float32)
        # np.save('./data/hk_fake_8_14/label/pose_mat_14.npy', pose_mat_14_data)

        # # use pose_mat to get pose_mat_fixed_landmarks
        # save_p = './data/hk_fake_8_14/label/pose_mat_fixed_landmarks.npy'
        # _ = HP.test_rt(save_p, mode=False)  # True, False 两者效果均错误, 世界坐标的旋转平移的逆变换能使得像素坐标物体不动?
    else:
        print("yy")
