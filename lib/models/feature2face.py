from turtle import forward
import torch 
import torch.nn as nn
from .feature2face_D import Feature2Face_D
from .feature2face_G import Feature2Face_G


class Feature2Face(nn.Module):          
    def __init__(self, cfg):
        """Initialize the Feature2Face class.

        Parameters:
            cfg: 
        """
        super(Feature2Face, self).__init__()
        self.cfg = cfg
     
        self.FFG = Feature2Face_G(cfg)
        self.FFD = Feature2Face_D(cfg)
        
    
    def forward(self, feature_map, cand_img):
        input_feature_maps = torch.cat([feature_map, cand_img], dim=1)
        fake_pred = self.FFG(input_feature_maps)
        
        return input_feature_maps, fake_pred
    
    def backward_d(self):
        
        pass