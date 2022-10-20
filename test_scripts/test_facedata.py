import sys
import os

p_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(p_dir)
sys.path.insert(0, p_dir)

import numpy as np
import cv2
from lib.utils.util import tensor2im
from lib.config.config import Config
from lib.utils.visisual import draw_face_feature_maps
from lib.datasets.face_dataset import FaceDatasetCustom

# from .video_preprocess import RT_compute


path = './configs/audio2feature_8_14.yaml'
cfg = Config.fromfile(path)

FD = FaceDatasetCustom(cfg)
x = FD[14100]
print(x.keys())

print(x['feature_map'].shape, x['cand_image'].shape, x['tgt_image'].shape, x['weight_mask'].shape, x['points'].shape)

# print(type(x['feature_map']))

mean_pts3d = np.load('./data/hk_fake_8_14/label/mean_pts_3d_fixed_countour_nose_closemouth.npy')   # (68, 3)
mean_pts2d = mean_pts3d[..., :2]*256 + 256

shift_landmarks = list(range(0, 17)) + list(range(27,36))

src = mean_pts2d[shift_landmarks, ...]
dst = x['points'][shift_landmarks, ...]

diff_xy = (dst - src).mean(0)

shifted_mean_pts2d = mean_pts2d + diff_xy

mean_landmark_img = draw_face_feature_maps(mean_pts2d)
shifted_mean_pts2d_img = draw_face_feature_maps(shifted_mean_pts2d)
cv2.imwrite('xxmean_landmark.png', mean_landmark_img)

fmap = tensor2im(x['feature_map'])
ftgi = tensor2im(x['tgt_image'])
# 原始特征图 + 原始图 + 平均图
im = (fmap[..., np.newaxis] + ftgi + mean_landmark_img[..., np.newaxis])/3

print(np.sum(mean_landmark_img==255), np.sum(fmap==255))

g1 = (mean_landmark_img==255)*255
g2 = (fmap == 255)*255
g3 = (shifted_mean_pts2d_img == 255) * 255

im3 = (g2 + g3) / 2


# im2 = (g1 + g2) / 2

cv2.imwrite('xxim3.png', im)

# cv2.imwrite('xxim2.png', im2)
cv2.imwrite('xximg3.png', im3)