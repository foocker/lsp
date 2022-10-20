import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_shoulder_points(img, shoulder_points):
    num = int(shoulder_points.shape[0] / 2)
    for i in range(2):
        for j in range(num - 1):
            pt1 = [int(flt) for flt in shoulder_points[i * num + j]]
            pt2 = [int(flt) for flt in shoulder_points[i * num + j + 1]]
            img = cv2.line(img, tuple(pt1), tuple(pt2), 255, 2)  # BGR
        
    return img

part_list = [[list(range(0, 17))],                               # contour
            [[17,18,19,20,21,17]],                               # right eyebrow
            [[22,23,24,25,26,22]],                               # left eyebrow
            [range(27, 36)],                                     # nose
            [[36,37,38,39], [39,40,41,36]],                      # right eye
            [[42,43,44,45], [45,46,47,42]],                      # left eye
            [range(48, 55), [54,55,56,57,58,59,48]],             # mouth
            [[60,61,62,63,64], [64,65,66,67,60]]                 # tongue
            ]
mouth_outer = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,59, 48]

def draw_face_feature_maps(keypoints, size=(512, 512)):
    w, h = size
    # edge map for face region from keypoints
    im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
    for edge_list in part_list:
        for edge in edge_list:
            for i in range(len(edge)-1):
                pt1 = [int(flt) for flt in keypoints[edge[i]]]
                pt2 = [int(flt) for flt in keypoints[edge[i + 1]]]
                im_edges = cv2.line(im_edges, tuple(pt1), tuple(pt2), 255, 2)

    return im_edges


def plot_verts(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (0, 0, 255)
    elif color == 'y':
        c = (0, 255, 255)
    image = image.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image,(int(st[0]), int(st[1])), 1, c, 2) 

    return image


def plot_kpts(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    if kpts.shape[0] == 73:
        end_list = np.array([15, 20, 26, 44, 57, 61], dtype=np.int32) # 73  貌似不连续
    elif kpts.shape[0] == 68:
        end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68, 61], dtype = np.int32) - 1  # 68
    else:
        raise ValueError(f'Not impleted for {kpts.shape[0]} points')
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()
    radius = max(int(min(image.shape[0], image.shape[1])/200), 1)
    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        if kpts.shape[1]==4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        if i in end_list:
            continue
        ed = kpts[i + 1, :2]
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), c, radius)  # 关键点顺序不同
        # image = cv2.circle(image,(int(st[0]), int(st[1])), radius, c, radius*2)  
        # image = cv2.putText(image, f'{i}', org=(int(st[0]), int(st[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2, color=c, thickness=1)  

    return image

def img2video(image_folder, video_name, img_format='png', fps=60.0):
    f = lambda x: float(x.split('_')[-1][:-4])
    images = [img for img in os.listdir(image_folder) if img.endswith(".{}".format(img_format))]
    images = sorted(images, key=f)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4v')

    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height), True)

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        video.write(img)

    video.release()

    print("Succeeds!")
    

def frames2video(video_name, frames, fps=60.0):
    '''
    frames should be iter sequence yield
    '''
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")

    # video = cv2.VideoWriter(video_name, fourcc, fps, (512, 512), True)
    video = cv2.VideoWriter(video_name, 0x7634706d, fps, (512, 512), True)
    for frame in frames:
        video.write(frame)
    video.release()
    
def lmark2img( lmark,img= None, c = 'w'):     
    preds = lmark
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 1, 1)
    #if img != None:
#         img  = io.imread(img_path)
    ax.imshow(img)
    ax.plot(preds[0:17,0],preds[0:17,1]  ,marker='o',markersize=1,linestyle='-',color=c,lw=1)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color=c,lw=1)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color=c,lw=1)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color= c,lw=1)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color= c,lw=1)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color= c,lw=1)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color= c,lw=1)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color= c,lw=1)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color= c,lw=1) 
    ax.axis('off')
    ax.set_xlim(ax.get_xlim()[::-1])
    
    return plt


def compare_vis(img,lmark1,lmark2):
    # img  = io.imread(img_path)
    preds = lmark1
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1) 
    ax.axis('off')
    
    preds = lmark2
    
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img)
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1) 
    ax.axis('off')
    
    return plt


Mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], \
         [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], \
         [66, 67], [67, 60]]

Nose = [[27, 28], [28, 29], [29, 30], [30, 31], [30, 35], [31, 32], [32, 33], \
        [33, 34], [34, 35], [27, 31], [27, 35]]

leftBrow = [[17, 18], [18, 19], [19, 20], [20, 21]]
rightBrow = [[22, 23], [23, 24], [24, 25], [25, 26]]

leftEye = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [36, 41]]
rightEye = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47]]

other = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], \
         [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], \
         [12, 13], [13, 14], [14, 15], [15, 16]]

faceLmarkLookup = Mouth + Nose + leftBrow + rightBrow + leftEye + rightEye + other

def plot_flmarks3D(pts, lab, xLim, yLim, zLim,rotate=False,  figsize=(10, 10), sentence =None):
    pts = np.reshape(pts, (68, 3))

    if pts.shape[0] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
        print (lookup)
    else:
        lookup = faceLmarkLookup

    plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    l, = ax.plot3D([], [], [], 'ko', ms=2)

    lines = [ax.plot([], [], [], 'k', lw=1)[0] for _ in range(3*len(lookup))]
    ax.set_xlim3d(xLim)     
    ax.set_ylim3d(yLim)     
    ax.set_zlim3d(zLim)
    ax.set_xlabel('x', fontsize=28)
    ax.set_ylabel('y', fontsize=28)
    ax.set_zlabel('z', fontsize=28)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    if rotate:
            angles = np.linspace(60, 120,1)
    else:
        angles = np.linspace(60, 60, 1)

    ax.view_init(elev=60, azim=angles[0])
    l.set_data(pts[:,0], pts[:,1])
    l.set_3d_properties(pts[:,2])
    cnt = 0
    for refpts in lookup:
        lines[cnt].set_data([pts[refpts[1], 0], pts[refpts[0], 0]], [pts[refpts[1], 1], pts[refpts[0], 1]])
        lines[cnt].set_3d_properties([pts[ refpts[1], 2], pts[refpts[0], 2]])
        cnt+=1
    if sentence is not None:
        plt.xlabel(sentence)
    plt.savefig(lab, dpi = 300, bbox_inches='tight')
    plt.clf()
    plt.close()
    
    
def draw_face_feature_maps(keypoints, size=(512, 512)):
    part_list = [[list(range(0, 17))],                               # contour
                [[17,18,19,20,21,17]],                               # right eyebrow
                [[22,23,24,25,26,22]],                               # left eyebrow
                [list(range(27, 36)) + [30]],                        # nose
                [[36,37,38,39], [39,40,41,36]],                      # right eye
                [[42,43,44,45], [45,46,47,42]],                      # left eye
                [range(48, 55), [54,55,56,57,58,59,48]],             # mouth
                [[60,61,62,63,64], [64,65,66,67,60]]                 # tongue
                ]
    w, h = size
    # edge map for face region from keypoints
    im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
    for index, edge_list in  enumerate(part_list):
        for edge in edge_list:
            for i in range(len(edge)-1):
                pt1 = [int(flt) for flt in keypoints[edge[i]]]
                pt2 = [int(flt) for flt in keypoints[edge[i + 1]]]
                im_edges = cv2.line(im_edges, tuple(pt1), tuple(pt2), 255, 2)
                # if index == 0:
                #     im_edges = cv2.putText(im_edges, f'{i}', org=tuple(pt1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2, color=(255, 0, 0), thickness=1)  

    return im_edges