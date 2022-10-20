import sys
import os

p_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, p_dir)

import torch 

from lib.lsp import LSP
from lib.config.config import Config
from lib.utils.util import tensor2im, save_image
from lib.datasets.face_dataset import FaceDatasetCustom


if __name__ == "__main__":
    path = './configs/audio2feature.yaml'
    cfg = Config.fromfile(path)
    lsp = LSP(cfg)
    facedata = FaceDatasetCustom(cfg)
    
    for i, fd in enumerate(facedata):
        feature_map, cand_img, tgt_img, facial_mask = fd['feature_map'], fd['cand_image'], fd['tgt_image'], fd['weight_mask']
    
        feature_map = feature_map.to(lsp.device)
        feature_map = torch.unsqueeze(feature_map, 0)
        cand_img = cand_img.to(lsp.device)
        cand_img = torch.unsqueeze(cand_img, 0)
        
        fake_pred = lsp.inference_g(feature_map, cand_img)
        pred_fake = tensor2im(fake_pred[0])
        tgt_img = tensor2im(tgt_img)
        feature_map = tensor2im(feature_map[0])
        if not os.path.exists(cfg.test_gan_dir):
            os.makedirs(cfg.test_gan_dir, exist_ok=True)
        save_image(pred_fake, os.path.join(cfg.test_gan_dir, f'test_{i}_generation.png'))
        save_image(tgt_img, os.path.join(cfg.test_gan_dir, f'tgt_{i}.png'))
        save_image(feature_map, os.path.join(cfg.test_gan_dir, f'feature_map_{i}.png'))
        
        if i == 200:
            break
