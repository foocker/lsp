# cp from https://github.com/lelechen63/Talking-head-Generation-with-Rhythmic-Head-Motion
from scipy.spatial.transform import Rotation as R
import numpy as np
import random
import math
import cv2
import copy

# Lookup tables for drawing lines between points
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


def mounth_open2close(lmark): 
    '''
    if the open rate is too large, we need to manually make the mounth to be closed.
    the input lamrk need to be (68,2 ) or (68,3)
    '''
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    # upper_part = [49,50,51,52,53]
    # lower_part = [59,58,57,56,55]
    diffs = []

    for k in range(3):
        mean = (lmark[open_pair[k][0],:2] + lmark[open_pair[k][1],:2] )/ 2
        diffs.append((mean - lmark[open_pair[k][0],:2]).copy())
        lmark[open_pair[k][0],:2] = mean - (mean - lmark[open_pair[k][0],:2]) * 0.3
        lmark[open_pair[k][1],:2] = mean + (mean - lmark[open_pair[k][0],:2]) * 0.3
    diffs.insert(0, 0.6 * diffs[2])
    diffs.append( 0.6 * diffs[2])
    diffs = np.asarray(diffs)
    lmark[49:54,:2] +=  diffs
    lmark[55:60,:2] -=  diffs
    
    return lmark

def mouth_pred_enhance(landmarks, enhance_ratio=1.5):
    pass

def get_roi(lmark):
    tempolate = np.zeros((256, 256 , 3), np.uint8)
    eyes =[17, 20 , 21, 22, 24,  26, 36, 39,42, 45]
    eyes_x = []
    eyes_y = []
    for i in eyes:
        eyes_x.append(lmark[i,0])
        eyes_y.append(lmark[i,1])
    min_x = lmark[eyes[np.argmin(eyes_x)], 0] 
    max_x = lmark[eyes[np.argmax(eyes_x)], 0] 
    min_y = lmark[eyes[np.argmin(eyes_y)], 1]
    
    max_y = lmark[eyes[np.argmax(eyes_y)], 1]
    min_x = max(0, int(min_x-10) )
    max_x = min(255, int(max_x+10) )
    min_y = max(0, int(min_y-10) )
    max_y = min(255, int(max_y+10) )

    tempolate[ int(min_y): int(max_y), int(min_x):int(max_x)] = 1 
    mouth = [48, 50, 51, 54, 57]
    mouth_x = []
    mouth_y = []
    for i in mouth:
        mouth_x.append(lmark[i,0])
        mouth_y.append(lmark[i,1])
    min_x2 = lmark[mouth[np.argmin(mouth_x)], 0] 
    max_x2 = lmark[mouth[np.argmax(mouth_x)], 0] 
    min_y2 = lmark[mouth[np.argmin(mouth_y)], 1]
    max_y2 = lmark[mouth[np.argmax(mouth_y)], 1] 

    min_x2 = max(0, int(min_x2-10) )
    max_x2 = min(255, int(max_x2+10) )
    min_y2 = max(0, int(min_y2-10) )
    max_y2 = min(255, int(max_y2+10) )

    tempolate[int(min_y2):int(max_y2), int(min_x2):int(max_x2)] = 1
    
    return  tempolate

def eye_blinking_o(lmark, rate = 40):
    '''
    lmark shape (k, 68,2) or (k,68,3)
    '''
    length = lmark.shape[0]
    bink_time = math.floor(length / float(rate))
    print('bink_time for eye blinking', bink_time, 'length is:', length)
    
    eyes =[[37,41],[38,40] ,[43,47],[44,46]]  # [upper, lower] , [left1, left2, right1, right1]
    
    for i in range(bink_time):
        for e in eyes:
            dis =  (np.abs(lmark[0, e[0],:2] -  lmark[0, e[1],:2] ) / 2)

            # -2 
            lmark[rate * (i + 1)-2, e[0],:2] += 0.45 * (dis)
            lmark[rate * (i + 1)-2, e[1],:2] -= 0.45 * (dis)
            # +2
            lmark[rate * (i + 1)+2, e[0], :2] += 0.45 * (dis)
            lmark[rate * (i + 1)+2, e[1], :2] -= 0.45 * (dis)

            # -1
            lmark[rate * (i + 1)-1, e[0], :2] += 0.85 * (dis)
            lmark[rate * (i + 1)-1, e[1], :2] -= 0.85 * (dis)
            # +1
            lmark[rate * (i + 1)+1, e[0], :2] += 0.8 * (dis)
            lmark[rate * (i + 1)+1, e[1], :2] -= 0.8 * (dis)

            # 0
            lmark[rate * (i + 1), e[0], :2] += 0.95 * (dis)
            lmark[rate * (i + 1), e[1], :2] -= 0.95 * (dis)
            
    return lmark

def eye_blinking_inverse(lmark, fps = 60, keep_second=7, mode='open'):
    '''
    lmark shape (k, 68,2) or (k,68,3), open eyes, mode: open, close
    '''
    length = lmark.shape[0]
    blink_time = math.floor(length / float(fps))
    
    blink_time_clip = round(blink_time / keep_second)
    
    blink_starts = sorted(random.sample(range(1, blink_time), blink_time_clip))
    print(blink_starts, 'blink_starts')
    
    line_space = np.linspace(0.05, 0.95, fps)
    
    print('bink_time for eye blinking', blink_time, 'length is:', length)
    
    eyes =[[37,41],[38,40] ,[43,47],[44,46]]  # [upper, lower] , [left1, left2, right1, right1]
    
    def op(a, b, mode=mode):
        if mode == 'open':
            return a + b
        return a - b
    
    for i in blink_starts:
        for j in range(fps):
            for e in eyes:
                dis_ij = -lmark[i*fps+j, e[0], 1] + lmark[i+j, e[1], 1]
                lmark[i*fps+j, e[0], 1] -= line_space[j] * dis_ij
                lmark[i*fps+j, e[1], 1] += line_space[j] * dis_ij
            # print('mean rate is: ', openrate_eye(lmark[i*fps+j]) , i*fps+j)  # 6~15
    return lmark

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2,:] *= -1
        R = Vt.T * U.T
    
    t = -R*centroid_A.T + centroid_B.T

    return R, t

def rt_to_degree(RT):
    #   RT (6,)
    RT = np.mat(RT)
    # recover the transformation
    rec = RT[0, :3]
    r = R.from_rotvec(rec)
    ret_R =  r.as_euler('zyx', degrees=True)
    
    return ret_R

def reverse_rt(source,  RT):
    #source (68,3) , RT (6,)
    source =  np.mat(source)
    RT = np.mat(RT)
    # recover the transformation
    rec = RT[0,:3]
    r = R.from_rotvec(rec)
    ret_R = r.as_dcm()
    ret_R2 = ret_R[0].T
    ret_t = RT[0,3:]
    ret_t = ret_t.reshape(3,1)
    ret_t2 = - ret_R2 * ret_t
    ret_t2 = ret_t2.reshape(3,1)
    A3 = ret_R2 *   source.T +  np.tile(ret_t2, (1,68))
    A3 = A3.T
    
    return A3


class faceNormalizer(object):
    # Credits: http://www.learnopencv.com/face-morph-using-opencv-cpp-python/
    w = 256
    h = 256

    def __init__(self, w = 256, h = 256):
        self.w = w
        self.h = h

    def similarityTransform(self, inPoints, outPoints):
        s60 = math.sin(60*math.pi/180)
        c60 = math.cos(60*math.pi/180)
      
        inPts = np.copy(inPoints).tolist()
        outPts = np.copy(outPoints).tolist()
        
        xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
        
        inPts.append([np.int(xin), np.int(yin)])
        
        xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
        
        outPts.append([np.int(xout), np.int(yout)])
        
        tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
        
        return tform

    def tformFlmarks(self, flmark, tform):
        transformed = np.reshape(np.array(flmark), (68, 1, 2))           
        transformed = cv2.transform(transformed, tform)
        transformed = np.float32(np.reshape(transformed, (68, 2)))
        return transformed

    def alignEyePoints(self, lmarkSeq):
        w = self.w
        h = self.h

        alignedSeq = copy.deepcopy(lmarkSeq)
        firstFlmark = alignedSeq[0,:,:]
        
        eyecornerDst = [ (np.float(0.3 * w ), np.float(h / 3)), (np.float(0.7 * w ), np.float(h / 3)) ]
        eyecornerSrc  = [ (firstFlmark[36, 0], firstFlmark[36, 1]), (firstFlmark[45, 0], firstFlmark[45, 1]) ]

        tform = self.similarityTransform(eyecornerSrc, eyecornerDst)

        for i, lmark in enumerate(alignedSeq):
            alignedSeq[i] = self.tformFlmarks(lmark, tform)

        return alignedSeq

    def alignEyePointsV2(self, lmarkSeq):
        w = self.w
        h = self.h

        alignedSeq = copy.deepcopy(lmarkSeq)
        
        eyecornerDst = [ (np.float(0.3 * w ), np.float(h / 3)), (np.float(0.7 * w ), np.float(h / 3)) ]
    
        for i, lmark in enumerate(alignedSeq):
            curLmark = alignedSeq[i,:,:]
            eyecornerSrc  = [ (curLmark[36, 0], curLmark[36, 1]), (curLmark[45, 0], curLmark[45, 1]) ]
            tform = self.similarityTransform(eyecornerSrc, eyecornerDst)
            alignedSeq[i,:,:] = self.tformFlmarks(lmark, tform)

        return alignedSeq

    def transferExpression(self, lmarkSeq, meanShape):
        exptransSeq = copy.deepcopy(lmarkSeq)
        firstFlmark = exptransSeq[0,:,:]
        indexes = np.array([60, 64, 62, 67])
        
        tformMS = cv2.estimateRigidTransform(firstFlmark[:,:], np.float32(meanShape[:,:]) , True)

        sx = np.sign(tformMS[0,0])*np.sqrt(tformMS[0,0]**2 + tformMS[0,1]**2)
        sy = np.sign(tformMS[1,0])*np.sqrt(tformMS[1,0]**2 + tformMS[1,1]**2)
        print (sx, sy)
        prevLmark = copy.deepcopy(firstFlmark)
        prevExpTransFlmark = copy.deepcopy(meanShape)
        zeroVecD = np.zeros((1, 68, 2))
        diff = np.cumsum(np.insert(np.diff(exptransSeq, n=1, axis=0), 0, zeroVecD, axis=0), axis=0)
        msSeq = np.tile(np.reshape(meanShape, (1, 68, 2)), [lmarkSeq.shape[0], 1, 1])

        diff[:, :, 0] = abs(sx)*diff[:, :, 0]
        diff[:, :, 1] = abs(sy)*diff[:, :, 1]

        exptransSeq = diff + msSeq

        return exptransSeq

    def unitNorm(self, flmarkSeq):
        normSeq = copy.deepcopy(flmarkSeq)
        normSeq[:, : , 0] /= self.w
        normSeq[:, : , 1] /= self.h
        return normSeq


def oned_smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
   
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise( ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': # moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y
 
def smooth(kps, ALPHA1=0.2, ALPHA2=0.7):

    n = kps.shape[0]

    kps_new = np.zeros_like(kps)

    for i in range(n):
        if i==0:
            kps_new[i,:,:] = kps[i,:,:]
        else:
            kps_new[i,:48,:] = ALPHA1 * kps[i,:48,:] + (1-ALPHA1) * kps_new[i-1,:48,:]
            kps_new[i,48:,:] = ALPHA2 * kps[i,48:,:] + (1-ALPHA2) * kps_new[i-1,48:,:]

    # np.save(out_file, kps_new)
    return kps_new


def plot_landmarks( landmarks):
    # landmarks = np.int32(landmarks)
    blank_image = np.zeros((256,256,3), np.uint8) 

    # cv2.polylines(blank_image, np.int32([points]), True, (0,255,255), 1)

    cv2.polylines(blank_image, np.int32([landmarks[0:17]]) , True, (0,255,255), 2)
 
    cv2.polylines(blank_image,  np.int32([landmarks[17:22]]), True, (255,0,255), 2)

    cv2.polylines(blank_image, np.int32([landmarks[22:27]]) , True, (255,0,255), 2)

    cv2.polylines(blank_image, np.int32([landmarks[27:31]]) , True, (255,255, 0), 2)

    cv2.polylines(blank_image, np.int32([landmarks[31:36]]) , True, (255,255, 0), 2)

    cv2.polylines(blank_image, np.int32([landmarks[36:42]]) , True, (255,0, 0), 2)
    cv2.polylines(blank_image, np.int32([landmarks[42:48]]) , True, (255,0, 0), 2)

    cv2.polylines(blank_image, np.int32([landmarks[48:60]]) , True, (0, 0, 255), 2)

    return blank_image


def crop_mouth(img, lmark):
    (x, y, w, h) = cv2.boundingRect(lmark[48:68,:-1].astype(int))

    center_x = x + int(0.5 * w)

    center_y = y + int(0.5 * h)

    r = 32
    new_x = center_x - r
    new_y = center_y - r
    roi = img[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
    return roi


def mse_metrix(lmark1, lmark2):
    #input shape (68,3)
    distance =  np.square(lmark1 - lmark2)
    if distance.shape == (68,3):
        return distance[:,:-1].mean()
    else:
        return distance.mean()
    

def openrate(lmark1):
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    open_rate1 = []
    for k in range(3):
        open_rate1.append(lmark1[open_pair[k][0],:2] - lmark1[open_pair[k][1], :2])
        
    open_rate1 = np.asarray(open_rate1)
    return open_rate1.mean()

def openrate_eye(Lmark):
    open_pair = [[37,41],[38,40] ,[43,47],[44,46]]
    open_rate = []
    for k in range(len(open_pair)):
        open_rate.append(Lmark[open_pair[k][1], 1] - Lmark[open_pair[k][0], 1])
    open_rate = np.array(open_rate)
    return open_rate.mean()


def openrate_metrix(lmark1, lmark2):
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    open_rate1 = []
    open_rate2 = []
    for k in range(3):
        open_rate1.append(lmark1[open_pair[k][0],:2] - lmark1[open_pair[k][1], :2])
        
        
        open_rate2.append(lmark2[open_pair[k][0],:2] - lmark2[open_pair[k][1], :2])
        
    open_rate1 = np.asarray(open_rate1)
    open_rate2 = np.asarray(open_rate2)
    
    return mse_metrix(open_rate1, open_rate2) 

