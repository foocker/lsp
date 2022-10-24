# path = './configs/hk_fake_38.yaml'
path = "./configs/audio2feature.yaml"


from lib.config.config import Config

cfg = Config.fromfile(path)

# print(cfg.model_params)
# print(cfg.filename)
# print(cfg.model_params.Image2Image)
# print(cfg.VERSION)

# from configs.audio2feature import get_parser_config

# cfg = DictAction(get_parser_config())

print(cfg.predict_length)
