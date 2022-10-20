from glob import glob
import os

from .base_dataset import BaseDataset
import os.path as osp
import torch
from skimage.io import imread
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from albumentations import pytorch as albp


class FaceDatasetCustom(BaseDataset):
    def __init__(self, cfg):
        '''
        if good and time is enought, it's best to change the Config cfg
        
        '''
        BaseDataset.__init__(self, cfg)
        self.state = 'Train' if cfg.isTrain else 'Test'
        self.img_dir = cfg.img_dir
        self.label_dir = cfg.label_dir
        self.img_candidates_dir = cfg.candidates
        self.dataset_name = osp.dirname(self.img_dir)
   
        # default settings for 68 points
        # currently, we have 8 parts for face parts
        self.part_list = [[list(range(0, 17))],                                # contour
                          [[17,18,19,20,21,17]],                               # right eyebrow
                          [[22,23,24,25,26,22]],                               # left eyebrow
                          [list(range(27, 36)) + [30]],                                     # nose
                          [[36,37,38,39], [39,40,41,36]],                      # right eye
                          [[42,43,44,45], [45,46,47,42]],                      # left eye
                          [range(48, 55), [54,55,56,57,58,59,48]],             # mouth
                          [[60,61,62,63,64], [64,65,66,67,60]]                 # tongue
                         ]
        self.mouth_outer = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,59, 48]
        self.label_list = [1, 1, 2, 3, 3, 4, 5] # labeling for different facial parts
        
        
        landmark_path = osp.join(self.label_dir, 'landmarks2d_original.npy')  # (18415, 1, 68, 2)
        landmark2D = np.load(landmark_path).astype(np.float32)
        landmark2D = landmark2D * 256 + 256
        # landmark2D[..., 0] = landmark2D[..., 0]* 256 + 256  # ndc
        # landmark2D[..., 1] = landmark2D[..., 1]* 256 + 256
        self.landmark2D = landmark2D

        img_list = glob(osp.join(self.img_dir, '*.png'))
        f = lambda x : float(osp.basename(x).split('.')[0].split('_')[-1])
        self.tgts_path = sorted(img_list, key=f)
            
        if not self.landmark2D.shape[0] == len(self.tgts_path):
            raise ValueError('In dataset {} length of landmarks and images are not equal!'.format(osp.dirname(self.img_dir)))
        
        # tracked 3d info 
        fit_data_path = osp.join(self.label_dir, '3d_fit_data.npz') 
        fit_data = np.load(fit_data_path)
        self.pts3d = fit_data['pts_3d'].astype(np.float32)
        self.rot = fit_data['rot_angles'].astype(np.float32)
        self.trans = fit_data['trans'].astype(np.float32)
        
        if not self.pts3d.shape[0] == len(self.tgts_path):
            raise ValueError('In dataset {} length of 3d pts and images are not equal!'.format(osp.dirname(self.img_dir)))
        
        # candidates images
        tmp = []
        candidates_imgs = os.listdir(self.img_candidates_dir)
        for j in candidates_imgs:
            output = imread(osp.join(self.img_candidates_dir,j))
            output = albp.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 'std':(0.5,0.5,0.5)})(image=output)['image']
            tmp.append(output)
        self.full_cand = torch.cat(tmp) 

        # shoulders 
        if cfg.shoulder:
            shoulder_path = osp.join(self.label_dir, 'shoulder_2D.npy')
            self.shoulder = np.load(shoulder_path, allow_pickle=True)
        else:
            self.shoulder = None
        
        # for only one
        self.sample_len = np.int32(np.floor((self.landmark2D.shape[0] - 60) / self.cfg.frame_jump) + 1)
        self.len = self.landmark2D.shape[0]
        self.sample_start = [0]
        self.total_len = 0
        self.total_len += self.sample_len
        
        self.image_pad = None   # all input image is 512*512
        
    
    def __getitem__(self, ind):
        
        data_index = ind*self.cfg.frame_jump + np.random.randint(self.cfg.frame_jump)
        
        target_ind = data_index + 1  # history_ind, current_ind
        landmarks = self.landmark2D[target_ind]  # [68, 2]
        if self.shoulder is not None:
            shoulders = self.shoulder[target_ind].copy()
        else:
            shoulders = None

            
        tgt_image = np.asarray(Image.open(self.tgts_path[target_ind]))  # input img is 512 and landmark is 512, so, remove resize, crop transform
        
        h, w, _ = tgt_image.shape
        
        ### transformations & online data augmentations on images and landmarks   
        self.get_crop_coords(landmarks, (w, h), self.dataset_name, random_trans_scale=0)  # 30.5 µs ± 348 ns  random translation    
              
        transform_tgt = self.get_transform(self.dataset_name, True, n_img=1, n_keypoint=1, flip=False)
        transformed_tgt = transform_tgt(image=tgt_image, keypoints=landmarks)

        tgt_image, points = transformed_tgt['image'], np.array(transformed_tgt['keypoints']).astype(np.float32)
        
        points = np.array(points, dtype=np.int32).reshape(68, 2)
        feature_map = self.get_feature_image(points, (self.cfg.loadSize, self.cfg.loadSize), shoulders, \
            self.image_pad)[np.newaxis, :].astype(np.float32)/255.
        feature_map = torch.from_numpy(feature_map)
        
        ## facial weight mask
        weight_mask = self.generate_facial_weight_mask(points, h, w)[None, :] 
        
        cand_image = self.full_cand  
        
        return_list = {'feature_map': feature_map, 'cand_image': cand_image, 'tgt_image': tgt_image, 'weight_mask': weight_mask, 'points': points}
           
        return return_list
    
    def generate_facial_weight_mask(self, points, h = 512, w = 512):
        mouth_mask = np.zeros([h, w, 1])  # w, h?
        points = points[self.mouth_outer]
        points = np.int32(points)
        mouth_mask = cv2.fillPoly(mouth_mask, [points], (255,0,0))
#        plt.imshow(mouth_mask[:,:,0])
        mouth_mask = cv2.dilate(mouth_mask, np.ones((45, 45))) / 255
        
        return mouth_mask.astype(np.float32)
    
    
    def get_transform(self, dataset_name, keypoints=False, n_img=1, n_keypoint=1, normalize=True, flip=False):
        min_x = getattr(self, 'min_x_' + str(dataset_name))
        max_x = getattr(self, 'max_x_' + str(dataset_name))
        min_y = getattr(self, 'min_y_' + str(dataset_name))
        max_y = getattr(self, 'max_y_' + str(dataset_name))
                
        additional_flag = False
        additional_targets_dict = {}
        if n_img > 1:
            additional_flag = True
            image_str = ['image' + str(i) for i in range(0, n_img)]
            for i in range(n_img):
                additional_targets_dict[image_str[i]] = 'image'
        if n_keypoint > 1:
            additional_flag = True
            keypoint_str = ['keypoint' + str(i) for i in range(0, n_keypoint)]
            for i in range(n_keypoint):
                additional_targets_dict[keypoint_str[i]] = 'keypoints'
        # TODO bug crop
        transform = A.Compose([
                # A.Crop(x_min=min_x, x_max=max_x, y_min=min_y, y_max=max_y),
                A.Resize(self.cfg.loadSize, self.cfg.loadSize),
                A.HorizontalFlip(p=flip),
                albp.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 'std':(0.5,0.5,0.5)} if normalize==True else None)],
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False) if keypoints==True else None,
                additional_targets=additional_targets_dict if additional_flag == True else None
                )
        
        return transform 
    
    def get_data_test_mode(self, landmarks, shoulder, pad=None):
        ''' get transformed data
        '''
       
        feature_map = torch.from_numpy(self.get_feature_image(landmarks, (self.cfg.loadSize, self.cfg.loadSize), \
            shoulder, pad)[np.newaxis, :].astype(np.float32)/255.)

        return feature_map 
    
    def get_feature_image(self, landmarks, size, shoulders=None, image_pad=None):
        # draw edges
        im_edges = self.draw_face_feature_maps(landmarks, size) 
        # cv2.imwrite('./xgtest41_im_edges.png', im_edges)
        if shoulders is not None:
            if image_pad is not None:
                top, bottom, left, right = image_pad
                delta_y = top - bottom
                delta_x = right - left
                shoulders[:, 0] += delta_x
                shoulders[:, 1] += delta_y
            im_edges = self.draw_shoulder_points(im_edges, shoulders)

        return im_edges


    def draw_shoulder_points(self, img, shoulder_points):
        num = int(shoulder_points.shape[0] / 2)
        for i in range(2):
            for j in range(num - 1):
                pt1 = [int(flt) for flt in shoulder_points[i * num + j]]
                pt2 = [int(flt) for flt in shoulder_points[i * num + j + 1]]
                img = cv2.line(img, tuple(pt1), tuple(pt2), 255, 2)  # BGR
            
        return img
    
    def draw_face_feature_maps(self, keypoints, size=(512, 512)):
        w, h = size
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
        for index, edge_list in  enumerate(self.part_list):
            for edge in edge_list:
                for i in range(len(edge)-1):
                    pt1 = [int(flt) for flt in keypoints[edge[i]]]
                    pt2 = [int(flt) for flt in keypoints[edge[i + 1]]]
                    im_edges = cv2.line(im_edges, tuple(pt1), tuple(pt2), 255, 2)
                    # if index == 0:
                    #     im_edges = cv2.putText(im_edges, f'{i}', org=tuple(pt1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2, color=(255, 0, 0), thickness=1)  

        return im_edges


    def get_crop_coords(self, keypoints, size, dataset_name, random_trans_scale=50): 
        # cut a rought region for fine cutting
        # here x towards right and y towards down, origin is left-up corner
        w_ori, h_ori = size
        min_y, max_y = keypoints[:,1].min(), keypoints[:,1].max()
        min_x, max_x = keypoints[:,0].min(), keypoints[:,0].max()                
        xc = (min_x + max_x) // 2
        yc = (min_y*3 + max_y) // 4  # 人脸原因？还是加了肩膀？
        h = w = min((max_x - min_x) * 2, w_ori, h_ori)
        
        if self.cfg.isTrain:
            # do online augment on landmarks & images
            # 1. random translation: move 10%
            x_bias, y_bias = np.random.uniform(-random_trans_scale, random_trans_scale, size=(2,))
            xc, yc = xc + x_bias, yc + y_bias
            
        # modify the center x, center y to valid position
        xc = min(max(0, xc - w//2) + w, w_ori) - w//2
        yc = min(max(0, yc - h//2) + h, h_ori) - h//2
        
        min_x, max_x = xc - w//2, xc + w//2
        min_y, max_y = yc - h//2, yc + h//2 
        
        setattr(self, 'min_x_' + str(dataset_name), int(min_x))
        setattr(self, 'max_x_' + str(dataset_name), int(max_x))
        setattr(self, 'min_y_' + str(dataset_name), int(min_y))
        setattr(self, 'max_y_' + str(dataset_name), int(max_y))


    def crop(self, img, dataset_name):
        min_x = getattr(self, 'min_x_' + str(dataset_name))
        max_x = getattr(self, 'max_x_' + str(dataset_name))
        min_y = getattr(self, 'min_y_' + str(dataset_name))
        max_y = getattr(self, 'max_y_' + str(dataset_name))    
        if isinstance(img, np.ndarray):
            return img[min_y:max_y, min_x:max_x]
        else:
            return img.crop((min_x, min_y, max_x, max_y))
  

    def __len__(self):  
        if self.cfg.isTrain:
            return self.total_len
        else:
            return 1

    def name(self):
        return 'FaceDatasetCustom'
        

