from turtle import Vec2D
from scipy.ndimage import gaussian_filter1d
import tqdm
import numpy as np
import math


def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)


def enhnace_mouth(landmark, scale):
    '''
    only one landmark, in 2d, here index is for 68 points, 3d is normalized to 0-1
    鼻子到上嘴唇,d1,下嘴唇到下巴d2,上嘴唇到上中间嘴唇保持相对距离不变,下同理,机嘴唇厚度
    变换幅度更小,嘴巴两边固定,或者变化幅度极小,上下嘴唇距离上当前简化为高斯分布变化.
    可增加采样频率,补齐更多嘴唇关键点. TODO
    '''
    # nose_down = landmark[32:35, 1]  # 鼻3点
    # mouth_out_up = landmark[48:55, 1]  # 上7
    # mouth_inner_up = landmark[61:64, 1]  # 中3
    # mouth_inner_down = landmark[65:68, 1]  # 中3
    # mouth_out_down = landmark[54:61, 1] # 下7
    # contour_down = landmark[7:10, 1] # 边缘3
    
    # d1 = np.mean(mouth_out_up[2:5]) - np.mean(nose_down)
    # d2 = np.mean(contour_down) - np.mean(mouth_out_down[2:5])
    
    # up_lift_bound = d1 / 8
    # down_lift_bound = d2 / 8
    
    # shift_bound = min(up_lift_bound, down_lift_bound)

    # mean, sigma = 0, 10
    # x = np.linspace(mean - 2.5*sigma, mean + 2.5*sigma, 11)
    # y = normal_distribution(x, mean, sigma)
    # y_seven = y[[0, 2, 4, 5, 6, 8, 10]] * scale*3
    # y_seven[0] = 0
    # y_seven[-1] = 0
    # if max(y_seven) > shift_bound:
    #     y_seven = y_seven - max(y_seven) + shift_bound
    # y_three = y_seven[[2, 3, 4]]
    
    # mouth_out_up -= y_seven
    # mouth_out_down += y_seven
    
    
    # mouth_inner_up -= y_three
    # mouth_inner_down += y_three

    
    # landmark[48:55, 1] = mouth_out_up
    # landmark[61:64, 1] = mouth_inner_up
    # landmark[65:68, 1] = mouth_inner_down
    # landmark[54:61, 1] = mouth_out_down
    base_shift = np.array([0, 2, 4, 6, 4, 2, 0]) * scale
    landmark[48:55, 1] -= base_shift
    landmark[61:64, 1] -= base_shift[2:5]
    landmark[65:68, 1] += base_shift[2:5]
    landmark[54:61, 1] += base_shift
    
    return landmark


def enhance_change(pred_feat):
    '''
    放大映射值本身
    '''
    pass


def serach_near(landmark_lab, landmark_pred):
    '''
    landmark_lab: (n, 68, 2) in view image space
    landmark_pred: (m, 68, 2) in view image space
    '''
    
    part_list = [[list(range(0, 17))],                                   # contour
                    [[17,18,19,20,21,17]],                               # right eyebrow
                    [[22,23,24,25,26,22]],                               # left eyebrow
                    [list(range(27, 36)) + [30]],                        # nose
                    [[36,37,38,39], [39,40,41,36]],                      # right eye
                    [[42,43,44,45], [45,46,47,42]],                      # left eye
                    [range(48, 55), [54,55,56,57,58,59,48]],             # mouth
                    [[60,61,62,63,64], [64,65,66,67,60]]                 # tongue
                ]
    

def landmark_smooth_3d(pts3d, smooth_sigma=0, area='only_mouth'):
    ''' smooth the input 3d landmarks using gaussian filters on each dimension.
    Args:
        pts3d: [N, points_num, 3]
    '''
    # per-landmark smooth
    points_num = pts3d.shape[1]
    if not smooth_sigma == 0:
        if area == 'all':
            pts3d = gaussian_filter1d(pts3d.reshape(-1, points_num*3), smooth_sigma, axis=0).reshape(-1, points_num, 3)
            
        elif area == 'only_mouth':
            if points_num == 68:
                mouth_pts3d = pts3d[:, 48:68, :].copy()
                mouth_pts3d = gaussian_filter1d(mouth_pts3d.reshape(-1, 20*3), smooth_sigma, axis=0).reshape(-1, 20, 3)
                pts3d = gaussian_filter1d(pts3d.reshape(-1, 68*3), smooth_sigma, axis=0).reshape(-1, 68, 3)
                pts3d[:, 48:68, :] = mouth_pts3d
            elif points_num == 73:
                mouth_pts3d = pts3d[:, 46:64, :].copy()
                mouth_pts3d = gaussian_filter1d(mouth_pts3d.reshape(-1, 18*3), smooth_sigma, axis=0).reshape(-1, 18, 3)
                pts3d = gaussian_filter1d(pts3d.reshape(-1, 73*3), smooth_sigma, axis=0).reshape(-1, 73, 3)
                pts3d[:, 46:64, :] = mouth_pts3d
            else:
                raise ValueError(f'Not implete for points number : {points_num}')
                
    return pts3d

def mouth_pts_AMP(pts3d, is_delta=True, method='XY', paras=[1,1]):
    ''' mouth region AMP to control the reaction amplitude.
    method: 'XY', 'delta', 'XYZ', 'LowerMore' or 'CloseSmall'
    '''
    points_num = pts3d.shape[1]
    if points_num == 73:
        ind_left = 46
        ind_right = 64
        lower_mouth = [53, 54, 55, 56, 57, 58, 59, 60]
        upper_mouth = [46, 47, 48, 49, 50, 51, 52, 61, 62, 63]
        
    elif points_num == 68:
        # 68
        ind_left = 48
        ind_right = 68
        lower_mouth = [54, 55, 56, 57, 58, 59, 60]
        upper_mouth = [48, 49, 50, 51, 52, 53, 54]
    else:
        raise ValueError(f'Not implemented for points {points_num}')
        
    if method == 'XY':
        AMP_scale_x, AMP_scale_y = paras
        if is_delta:
            pts3d[:, ind_left:ind_right, 0] *= AMP_scale_x
            pts3d[:, ind_left:ind_right, 1] *= AMP_scale_y
        else:
            mean_mouth3d_xy = pts3d[:, ind_left:ind_right, :2].mean(axis=0)
            pts3d[:, ind_left:ind_right, 0] += (AMP_scale_x-1) * (pts3d[:, ind_left:ind_right, 0] - mean_mouth3d_xy[:,0])
            pts3d[:, ind_left:ind_right, 1] += (AMP_scale_y-1) * (pts3d[:, ind_left:ind_right, 1] - mean_mouth3d_xy[:,1])
    elif method == 'delta':
        AMP_scale_x, AMP_scale_y = paras
        if is_delta:
            diff = AMP_scale_x * (pts3d[1:, ind_left:ind_right] - pts3d[:-1, ind_left:ind_right])
            pts3d[1:, ind_left:ind_right] += diff
    
    elif method == 'XYZ':
        AMP_scale_x, AMP_scale_y, AMP_scale_z = paras
        if is_delta:
            pts3d[:, ind_left:ind_right, 0] *= AMP_scale_x
            pts3d[:, ind_left:ind_right, 1] *= AMP_scale_y
            pts3d[:, ind_left:ind_right, 2] *= AMP_scale_z
    
    elif method == 'LowerMore':
        upper_x, upper_y, upper_z, lower_x, lower_y, lower_z = paras
        if is_delta:
            pts3d[:, upper_mouth, 0] *= upper_x
            pts3d[:, upper_mouth, 1] *= upper_y
            pts3d[:, upper_mouth, 2] *= upper_z
            pts3d[:, lower_mouth, 0] *= lower_x
            pts3d[:, lower_mouth, 1] *= lower_y
            pts3d[:, lower_mouth, 2] *= lower_z
            
    elif method == 'CloseSmall':
        open_x, open_y, open_z, close_x, close_y, close_z = paras
        nframe = pts3d.shape[0]
        for i in tqdm(range(nframe), desc='AMP mouth..'):
            if sum(pts3d[i, upper_mouth, 1] > 0) + sum(pts3d[i, lower_mouth, 1] < 0) > 16 * 0.3:
                # open
                pts3d[i, ind_left:ind_right, 0] *= open_x
                pts3d[i, ind_left:ind_right, 1] *= open_y
                pts3d[i, ind_left:ind_right, 2] *= open_z
            else:
                # close
                pts3d[:, ind_left:ind_right, 0] *= close_x
                pts3d[:, ind_left:ind_right, 1] *= close_y
                pts3d[:, ind_left:ind_right, 2] *= close_z
    
    return pts3d

def solve_intersect_mouth(pts3d):
    ''' solve the generated intersec lips, usually happens in mouth AMP usage.
    Args:
        pts3d: [N, points_num, 3]
    '''
    points_num = pts3d.shape[1]
    if points_num == 73:
        upper_outer_lip = list(range(47, 52))
        upper_inner_lip = [63, 62, 61]
        lower_inner_lip = [58, 59, 60]
        lower_outer_lip = list(range(57, 52, -1))
        
    elif points_num == 68:
        # 68
        upper_outer_lip = list(range(48, 55))
        upper_inner_lip = [63, 62, 61]
        lower_inner_lip = [65, 66, 67]
        lower_outer_lip = list(range(54, 61, -1))
    else:
        raise ValueError(f'Not implemented for points {points_num}')
    
    upper_inner = pts3d[:, upper_inner_lip]
    lower_inner = pts3d[:, lower_inner_lip]
    
    lower_inner_y = lower_inner[:,:,1]
    upper_inner_y = upper_inner[:,:,1]
    # all three inner lip flip
    flip = lower_inner_y > upper_inner_y
    flip = np.where(flip.sum(axis=1) == 3)[0]
    
    # flip frames
    inner_y_diff = lower_inner_y[flip] - upper_inner_y[flip]
    half_inner_y_diff = inner_y_diff * 0.5
    # upper inner
    pts3d[flip[:,None], upper_inner_lip, 1] += half_inner_y_diff
    # lower inner
    pts3d[flip[:,None], lower_inner_lip, 1] -= half_inner_y_diff
    # upper outer
    pts3d[flip[:,None], upper_outer_lip, 1] += half_inner_y_diff.mean()
    # lower outer
    pts3d[flip[:,None], lower_outer_lip, 1] -= half_inner_y_diff.mean()
    
    
    return pts3d

def headpose_smooth(headpose, smooth_sigmas=[0,0], method='gaussian'):
    rot_sigma, trans_sigma = smooth_sigmas
    rot = gaussian_filter1d(headpose.reshape(-1, 6)[:,:3], rot_sigma, axis=0).reshape(-1, 3)
    trans = gaussian_filter1d(headpose.reshape(-1, 6)[:,3:], trans_sigma, axis=0).reshape(-1, 3)
    headpose_smooth = np.concatenate([rot, trans], axis=1)

    return headpose_smooth

def similary_match_contour():
    '''
    因训练数据生成的效果比较好,因此将预测的contour做best match
    in the training landmark, 不靠谱,因训练数据做的处理,导致
    the pred contour almost same. so it's best use headpose
    the get the transformation about rotation and shift.
    '''
    pass