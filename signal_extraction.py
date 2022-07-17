import time
import numpy as np
import os
import cv2
import dlib

# load dlib face detector
from AdaptiveWeightNetwork.utils import bandpass_filter
from utils.util_api import bandpass_61

detector = dlib.get_frontal_face_detector()
# download the shape_predictor_68_face_landmarks.dat to face_detector_path folder
face_detector_path=''
predictor = dlib.shape_predictor(face_detector_path+'shape_predictor_68_face_landmarks.dat')
def extract_raw_signals(frames_path,video_num,ring_num):
    start_time=time.time()
    all_avgpixels1 = []
    all_avgpixels2 = []
    for n in range(1, video_num):
        avgpixels1 = []
        avgpixels2 = []
        frame_list = os.listdir(frames_path+str(n))
        frame_list.sort(key=lambda x: int(x.split('.')[0]))
        p=0
        for frame in frame_list:
            p=p+1
            if p<=200:
                frame_path = os.path.join(frames_path +str(n), frame)

                image = cv2.imread(frame_path)
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                rects = detector(image_gray, 0)
                landmarks = np.matrix([[p.x, p.y] for p in predictor(image_gray, rects[i]).parts()])
                landmark_4 = np.array(landmarks[4])  
                landmark_39 = np.array(landmarks[39])

                left_distance = np.sqrt(np.power((np.squeeze(landmark_4)[0] - np.squeeze(landmark_39)[0]),2)
                    + np.power((np.squeeze(np.array(landmark_4))[1]) - np.squeeze(np.array(landmark_39))[1],2))
                # left_distance = np.sqrt(np.power((np.squeeze(left_point)[0] - np.squeeze(right_point)[0]),2)
                #     + np.power((np.squeeze(np.array(left_point))[1]) - np.squeeze(np.array(right_point))[1],2))
                left_center = tuple(np.sequeeze(((landmark_4 + landmark_39) / 2).astype(int)))
                left_radius = int(left_distance / 2)

                for i in range(ring_num):
                    left_roi = np.full(image.shape[0:2] + (1,), 0, dtype=np.float32)
                    left_radius1 = int(left_radius / 2 * (ring_num - i))
                    left_radius2 = int(left_radius / 2 * (ring_num - 1 - i))
                    print(left_radius1)
                    print(left_radius2)
                    cv2.circle(left_roi, left_center, left_radius1, (255, 255, 255), -1)
                    if i != ring_num - 1:
                        cv2.circle(left_roi, left_center, left_radius2, (0, 0, 0), -1)
                    left_roi = np.int8(left_roi)

                    image = cv2.imread(frame_path)
                    ring = cv2.bitwise_and(image, image, mask=left_roi)
                    if i != ring_num - 1:
                        area = np.pi * (left_radius1 * left_radius1 - left_radius2 * left_radius2)
                        B, G, R = cv2.split(ring)
                        avgpixels1.append(np.sum(G) / area)
                    else:
                        area = np.pi * (left_radius1 * left_radius1)
                        B, G, R = cv2.split(ring)
                        avgpixels2.append(np.sum(G) / area)
                
        bandpass_filter()
        avgpixels[0]=bandpass_61(avgpixels[0])
        plt.plot(avgpixels[0][305:305+610])
        plt.show()
        avgpixels[1] = bandpass_61(avgpixels[1])
        plt.plot(avgpixels[1][305:305+610])
        plt.show()
        avgpixels[2] = bandpass_61(avgpixels[2])
        plt.plot(avgpixels[2][305:305+610])
        plt.show()
        all_avgpixels1.append(avgpixels1)
        all_avgpixels2.append(avgpixels2)
    np.savez('left_ring_data',all_avgpixels1,all_avgpixels2)
    end_time=time.time()
    print(end_time-start_time)
# get_ring_data_npz(2)


