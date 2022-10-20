from .audiovisual_dataset import AudioVisualDataset
from .face_dataset import FaceDatasetCustom


def build_dataset(cfg):
    if cfg.task_block == 'audio':
        dataset = AudioVisualDataset(cfg)
    elif cfg.task_block == 'face':
        dataset = FaceDatasetCustom(cfg)
        
    return dataset