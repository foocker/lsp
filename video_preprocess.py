import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import subprocess


def parser_json(jp):
    kp = []
    with open(jp, 'r') as f:
        data = json.load(f)
    shapes = data['shapes']
    for p in shapes:
        ps = p['points'][0]
        kp.append(ps)

    return kp
        
def img2video(image_folder, video_name, img_format='png'):
    f = lambda x: float(x.split('_')[-1][:-4])  # change by name
    images = [img for img in os.listdir(image_folder) if img.endswith(".{}".format(img_format))]
    images = sorted(images, key=f)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4v')

    video = cv2.VideoWriter(video_name, fourcc, 60.0, (width, height), True)

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        video.write(img)

    video.release()

    print("Succeeds!")
    

def lucas_kanade_method_imgs(resources_path,  kp=None, save_kp_dir='./'):

    if kp is None:
        raise ValueError(f'{kp} should not be None, using labelme to get the keypoints what u need on first frame')

    save_kps = []
    if not os.path.exists(save_kp_dir):
        os.makedirs(save_kp_dir, exist_ok=True)

    imgs = [img for img in os.listdir(resources_path) if img.endswith(".{}".format('png'))]
    f = lambda x: float(x.split('.')[0].split('_')[-1])
    imgs = sorted(imgs, key=f)
    print(imgs[:30])
    frame_first = cv2.imread(os.path.join(resources_path, imgs[0]))

    # Take first frame and find corners in it
    old_gray = cv2.cvtColor(frame_first, cv2.COLOR_BGR2GRAY)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # append the first kp
    kp_array = []
    for p in kp:
        kp_array.append(np.array(p).reshape(1, 2))
    p0 = np.array(kp_array, dtype=np.float32)
    save_kps.append(p0.reshape(-1, 2))

    mask = np.zeros_like(frame_first)
    # Create a mask image for drawing purposes
    color = [0, 0, 255]
    img_nums = len(imgs)

    save_randome = np.random.choice(range(img_nums), 10, replace=False)
    print(save_randome)
    # save_randome = random.sample(range(img_nums), 10)
    # change_frame = [750, 800, 1520, 1600, 1800, 2300, 2472, 2700]   # 视频中间切帧
    change_frame = None
    for i, img in tqdm(enumerate(imgs[1:])):
        frame_after = cv2.imread(os.path.join(resources_path, img))
        # frame = frame_after.copy()
        if change_frame is not None and i+1 in change_frame:
            p0 = parser_json(f'./crop_out_512_{i+1}.json')
            p0 = np.array(p0, dtype=np.float32).reshape(-1, 1, 2)

        frame_gray = cv2.cvtColor(frame_after, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if i in save_randome:
            # Draw the tracks
            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = list(map(int, [a, b, c, d]))
                mask = cv2.line(mask, (a, b), (c, d), color, 2)
                frame = cv2.circle(frame_after, (a, b), 5, color, -1)

                # Display the demo
                img_draw = cv2.add(frame, mask)
                cv2.imwrite(os.path.join(save_kp_dir, f'save_random_kp_{i}.png'), img_draw)
                # cv2.imshow("frame", img_draw)
                # k = cv2.waitKey(25) & 0xFF
                # if k == 27:
                #     break

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        save_kps.append(good_new)
        p0 = good_new.reshape(-1, 1, 2)
    print(len(save_kps))

    npy_save_path =os.path.join(save_kp_dir, 'shoulder_2D.npy')
    # if not os.path.exists(npy_save_path):
    np.save(npy_save_path, save_kps)

    
def check_should2d(shoulder_npy, imgs_dir):
    imgs = [img for img in os.listdir(imgs_dir) if img.endswith(".{}".format('png'))]
    f = lambda x: float(x.split('.')[0].split('_')[-1])
    imgs = sorted(imgs, key=f)
    shoulder = np.load(shoulder_npy, allow_pickle=True)
    if len(imgs) != shoulder.shape[0]:
        raise ValueError(f'{len(imgs)}, {shoulder.shape[0]}')
    for i, im in enumerate(imgs):
        img = cv2.imread(os.path.join(imgs_dir, im))
        points = shoulder[i].astype(np.int16)
        pint = points
        # pint = list(map(int, points))
        # mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 2)
        for p in pint:
            frame = cv2.circle(img, (p[0], p[1]), 5, (0, 0, 255), -1)
        frame = cv2.putText(frame, f'{i}', org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1) 
        cv2.imshow("frame", frame)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break


class FAN(object):
    '''
    获取视频的基本信息,格式互转,fps互转,音频提取,以人物为中心,做视频裁剪,视频切割,crop,size等.
    crop can reference
    https://superuser.com/questions/547296/resizing-videos-with-ffmpeg-avconv-to-fit-into-static-sized-player/1136305#1136305
    ffmpeg -i input -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:-1:-1:color=black" output
    '''
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def detect_img(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'
        
    
    def draw_retangle(self, img, bbox):
        l, t, r, b = list(map(int, bbox))
        
        img = cv2.rectangle(img, (l, t), (r,b), (0, 255, 0),cv2.LINE_4, 2)
        cv2.imwrite("random_face.png", img)
        
    def detect_video(self, vp):
        '''
        每隔2秒采样,检测视频所有被采样的人脸,并计算出所有人脸的最小外围框,以此确定视频需要被裁剪的边界.
        '''
        cap = cv2.VideoCapture(vp)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度
        frame_count = 0
        bboxes = []
        draw_one = True
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame_count += 1
                if frame_count % fps == 2:
                    bbox, _ = self.detect_img(frame)
                    if draw_one:
                        self.draw_retangle(frame.copy(), bbox)
                        draw_one = False
                    bboxes.append(bbox)
            else:
                break
            
        bboxes = np.array(bboxes)
        l, t = np.min(bboxes[:, 0]), np.min(bboxes[:, 1])
        r, b = np.max(bboxes[:, 2]), np.max(bboxes[:, 3])
        
        cap.release()
        # cv2.destroyAllWindows()  # cvDestroyAllWindows
        
        return [l, t, r, b], (width, height)
    
    
    def crop_video_letterbox(self, vp, keep_size=(512, 512), face_head_scale=1.5):
        '''
        在视频本身不切屏的情况下,包含人体,并裁剪到固定大小,宽度太大,则resize整个视频,若最长宽度太小,则对裁剪结果做padd效果
        '''
        crop_vp = vp
        [l, t, r, b], (w, h) = self.detect_video(vp)
        # here can write more complicate func
        wx_sacled = (r - l ) * face_head_scale
        hy_scaled = (b - t) * face_head_scale
        
        vp_dir = os.path.dirname(vp)
        scaled_vp  = os.path.join(vp_dir , 'half_' +  os.path.basename(vp))
        if max(wx_sacled, hy_scaled) > keep_size[0]:
            cmd = f'ffmpeg -i {vp} -strict -2 -vf scale=iw/2:ih/2 {scaled_vp}'
            print(cmd)
            subprocess.call([cmd], shell=True)
        if os.path.exists(scaled_vp):
            wx_sacled //= 2
            hy_scaled //= 2
            l //= 2
            t //= 2
            crop_vp = scaled_vp
        
        if max(wx_sacled, hy_scaled) > keep_size[0]:
            raise ValueError('common box in video of face is so large, try to get the data by hand!')
        
        # crop original video by keep size to get the needed video
        pad_x = (keep_size[0] - wx_sacled)// 2
        pad_y = (keep_size[1] - hy_scaled) // 2
        position = (max(l - pad_x, 0), max(0, t - pad_y))  # can add w, h check
        croped_vp = os.path.join(vp_dir, 'croped_' + os.path.basename(crop_vp))
        print(croped_vp)
        
        print(pad_x, pad_y, position)
        
        cmd = f'ffmpeg -i {crop_vp} -strict -2 -vf crop=w={keep_size[0]}:h={keep_size[1]}:x={position[0]}:y={position[1]} {croped_vp}'
        subprocess.call(cmd, shell=True)
        print('croped successed!')
        return croped_vp
        
    def video_read_message(self, vp):
        if not os.path.exists(vp):
            return 0
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            return 0

        fileSize = os.path.getsize(vp) / (2 ** 20)  # 单位Mib
        # fileSize = os.path.getsize(path_video) / (10**6)  # 单位Mb
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度
        vcodec = int(cap.get(cv2.CAP_PROP_FOURCC))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 总图像帧数

        video_message = {"vcode":vcodec, 'filesize': f"{fileSize} Mib",'fps':fps,'w':width,'h':height,'frame_count':total_frames}
        print(f"视频基本信息 = {video_message}")
        
        return video_message

    # ffmpeg.exe -i video/kanghui_5.mp4 -r 60 -acodec aac -vcodec h264 out.mp4
    def change_vd_fps(self, vd):
        name = os.path.basename(vd).split(".")[0]
        print(name, os.path.abspath(vd))
        # ffmpeg -i ./video/kanghui_5.mp4  -r 60 -acodec aac -vcodec h264 ./video/kanghui5__60fps.mp4
        result_file = os.path.join(os.path.dirname(vd), f'{name}_60fps.mp4')
        cmd = f'ffmpeg -i {vd} -strict -2 -r 60 -acodec aac -vcodec h264 {result_file}'
        subprocess.call([cmd], shell=True)
        return result_file


    def extract_audio(self, vp, sr=16000):
        name = os.path.basename(vp).split('.')[0]
        save_audio_dir = os.path.dirname(vp)
        cmd = f'ffmpeg -i {vp} -f wav -ar {sr} {os.path.join(save_audio_dir, name)}.wav'
        cmd_v = f'ffmpeg -i {vp} -vcodec copy -an {os.path.join(save_audio_dir, "noaudio_" + name)}.mp4'
        subprocess.call([cmd, cmd_v], shell=True)
        # subprocess.call([cmd_v], shell=True)
        
        print("Successed")

    def merge_audio_video(self, vd, wav):
        '''
        合并视频和音频
        '''
        name = os.path.basename(vd).split('.')[0]
        save_audio_dir = os.path.dirname(vd)
        cmd = f'ffmpeg -i {vd} -i {wav} -strict -2 {os.path.join(save_audio_dir, name+"_audio")}.mp4'

        subprocess.call([cmd], shell=True)
        print("Successed")

    def clip_video(self):
        '''
        主要是嘴巴和视频对齐问题
        '''
        # ffmpeg.exe -ss 180 -t 300 -accurate_seek -i kh_fake.mp4  -c copy  -avoid_negative_ts 1 hk_fake_38.mp4
        # ffmpeg -ss 10 -t 15 -i test.mp4 -c:v libx264 -c:a aac -strict experimental -b:a 98k cut.mp4 -y
        pass
    
    def extract_imgs(self, vp, dst_dir):
        # name = os.path.basename(vp).split('.')[0]
        # cmd = f'ffmpeg -i {vp} -strict -2  -filter:v fps=60 {os.path.join(dst_dir, name)}_%0d.png'
        cmd = f'ffmpeg -i {vp} -strict -2  -filter:v fps=60 {dst_dir}/%0d.png'
        subprocess.call([cmd], shell=True)
        print("Successed")
        bash = f'ls {dst_dir} | wc -l'
        subprocess.call([bash], shell=True)
        
    def video_preprocess(self):
        '''
        handle all
        '''
        # self.crop_video_letterbox()
        # self.change_vd_fps()
        # self.extract_audio()
        # self.extract_imgs()
        print("Successed")
        

if __name__ == "__main__":
    # vp = './hk_fake_38_half.mp4'
    # vp = './croped_hk_fake_38_half_60fps.mp4'
    # vp = './data/hk_fake_38/test_mouth_new.mp4'
    # vp = './data/hk_fake_38/original_landmarks.mp4'
    # vp = './data/hk_fake_38/landmarks3d_fixed_countour_nose.mp4'
    # vp = './data/hk_fake_38/test_all_train_add_headpose.mp4'
    croped_vp = './data/hk_fake_8_14/croped_half_hk_fake_8_14_60fps.mp4'
    dst_dir = '/home/yourname/DECA/DECA/hk_fake_8_14_imgs'
    
    # wav = './data/hk_fake_38/video_audio/audio.wav'
    # wav = './data/hk_fake_38/video_audio/Mayun.wav'
    
    # img2video('./imgs', './video/kanghui_5_10.avi')
    # kp = parser_json('./kanghui_5_1.json')
    # kp = parser_json('./kanghui_5_60fps_crop_512_1.json')
    # print(kp)
    # lucas_kanade_method_imgs('./kanghui_imgs_512', kp, save_kp_dir='./kp_save_kh')
    # check_should2d('./kp_save_kh/shoulder_2D.npy', './kanghui_imgs_512')
    # x = np.load('./kp_save_girl/shoulder_2D.npy')
    # print(x.shape)

    fan = FAN()
    # croped_vp = fan.crop_video_letterbox(vp, face_head_scale=1.5)
    # fan.video_read_message(croped_vp)
    # result_file = fan.change_vd_fps(croped_vp)
    # fan.video_read_message(result_file)
    # fan.video_read_message(croped_vp)
    fan.extract_imgs(croped_vp, dst_dir)
    
    # fan.extract_audio(croped_vp)
    # fan.merge_audio_video(vp, wav)
    # merge_audio_video('noaudio_croped_hk_fake_38_half_60fps.mp4', 'croped_hk_fake_38_half_60fps.wav')