from lib.solover.trainer import Trainer
from lib.config.config import Config
# from lib.utils.config import cfg
from lib.lsp import LSP
from torch.backends import cudnn


def main(cfg):
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    lsp = LSP(cfg)
    # a2f = Audio2Feature(cfg)
    trainer = Trainer(model=lsp, config=cfg)
    trainer.fit()


if __name__ == "__main__":
    path = './configs/audio2feature_8_14.yaml'
    cfg = Config.fromfile(path)
    # print(cfg)
    
    # a2f = Audio2Feature(cfg)
    # print(a2f)
    main(cfg)