# sh deca
# 可视化预测的2D landmank、3D landmank(红色表示不可见点)、粗略几何图形、详细几何图形和深度
# python demos/demo_reconstruct.py -i TestSamples/examples/newk --saveDepth True --saveObj True --saveKpt True --saveImages True --saveMat True
# /root/.cache/torch/hub/checkpoints/2DFAN4_1.6-c827573f02.zip

# 表情迁移（expression transfer）
# python demos/demo_transfer.py

# ffmpeg -i noaudio_croped_hk_fake_38_half_60fps.mp4 -strict -2  -filter:v fps=60 ./hk_fake_38_imgs/%0d.png
# 生成LSP数据
video=croped_half_hk_fake_8_14_60fps.mp4
python demos/lsp_data_generator.py  --video_path=./${video} --video_img_save=./hk_fake_8_14_imgs --save_audio_dir=./ -i ./hk_fake_8_14_imgs -s hk_fake_8_14_LSP/label --render_orig True --saveKpt True
