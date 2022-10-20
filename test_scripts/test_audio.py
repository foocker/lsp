import sys
import os

p_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, p_dir)


import librosa
from lib.models.networks import APC_encoder
from lib.audio_module.utils import compute_mel_one_sequence
from lib.datasets.audiovisual_dataset import AudioVisualDataset

from lib.config.config import Config


if __name__ == "__main__":
    print("test audio")
    # audio_path = '/home/yourname/lsp/data/hk_fake_38/video_audio/audio.wav'
    # audio, _ = librosa.load(audio_path, sr=16000)
    # mel80s = compute_mel_one_sequence(audio)
    cfg_path = os.path.join(p_dir, "configs/audio2feature_8_14.yaml")
    cfg = Config.fromfile(cfg_path)
    AVD = AudioVisualDataset(cfg)
    A2Lsamples, target_pts3d = AVD[10]
    print(A2Lsamples, target_pts3d)

