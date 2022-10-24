# lsp
recreating the LSP(LiveSpeechPortraits) alg.

## test config 
```
python test_config.py
```

## data preprocess
training data structure:
data
-- video_name
  -- APC
    --xx_APC_feature_xxx.npy
  --checkpoints
  --imgs
  --candidates
  --g_sample
  --label
    --3d_fit_data.npz
    --mean_pts_3d.npy
    --...
  --video_audio

```
python video_preprocess.py
1. video preprocess
    1. change fps to 60
    2. crop video to (512, 512)
    3. translate video to images
    4. extract audio from croped video
    5. merge audio, images, or video to new video.
2. translated imgs
    1. using deca to extract useful information 
    2. cd otherlib/deca
    sh run.sh

```
## train
change the paramertes of yaml config file:
task, task_block and some other parameters from model and some basic traning paramertes: max_epochs, log_steps, checkpoint_steps, n_epochs
```
python train.py
```

# infer
change the config as in train,
infer_a(audio), infer_h(headpose), infer_g(generator), infer_file_x, test_audio
```
python demo.py
```

## test
1. test config
2. test generator
3. test from test_mode(xx.yaml config from traing config), there are many combination
```
3. python demo.py
2. cd test_scripts
python test_xx.py
```

## improve
1. audio2landmark, data process, like landmarks aligement you can try:
[noise_resilient_3dtface](https://github.com/eeskimez/noise_resilient_3dtface) or others
2. change other 3d face reconstruction alg.
