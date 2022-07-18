import time
import numpy as np
import os
import cv2
import dlib

# load dlib face detector
from AdaptiveWeightNetwork.utils import bandpass_filter


detector = dlib.get_frontal_face_detector()
# download the shape_predictor_68_face_landmarks.dat to face_detector_path folder
face_detector_path=''
predictor = dlib.shape_predictor(face_detector_path+'shape_predictor_68_face_landmarks.dat')
signals_save_path=''
def extract_raw_signals(video_folder,signals_save_path,step_size):
    all_signals=[]
    video_list = os.listdir(video_folder)
    video_list.sort(key=lambda x: int(x))
    for n in range(1, len(video_list)):
        signals=[]
        l=0
        cap = cv2.VideoCapture(video_folder + video_list[n])  # 注意路径是否写对
        while (1):
            ret, image = cap.read()
            if ret == True and l<=1500-step_size:
                left_avgpixels1 = []
                left_avgpixels2 = []
                right_avgpixels1 = []
                right_avgpixels2 = []
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                rects = detector(image_gray, 0)
                landmarks = np.matrix([[p.x, p.y] for p in predictor(image_gray, rects[i]).parts()])
                landmark_4 = np.array(landmarks[4])
                landmark_39 = np.array(landmarks[39])
                landmark_33 = np.array(landmarks[33])
                landmark_12 = np.array(landmarks[12])
                landmark_42 = np.array(landmarks[42])

                left_center = tuple(np.sequeeze(((landmark_4 + landmark_39) / 2).astype(int)))
                left_radius = np.sqrt(np.power((left_center[0] - np.squeeze(landmark_33)[0]),2)
                                      + np.power(left_center[1] - np.squeeze(np.array(landmark_33))[1],2))/2

                right_center = tuple(np.sequeeze(((landmark_12 + landmark_42) / 2).astype(int)))
                right_radius = np.sqrt(np.power((right_center[0] - np.squeeze(landmark_33)[0]), 2)
                                      + np.power(right_center[1] - np.squeeze(np.array(landmark_33))[1], 2)) / 2

                for i in range(2):
                    left_roi = np.full(image.shape[0:2] + (1,), 0, dtype=np.float32)
                    right_roi = np.full(image.shape[0:2] + (1,), 0, dtype=np.float32)
                    left_radius1 = int(left_radius / 2 * (2 - i))
                    left_radius2 = int(left_radius / 2 * (1 - i))
                    right_radius1 = int(right_radius / 2 * (2 - i))
                    right_radius2 = int(right_radius / 2 * (1 - i))

                    cv2.circle(left_roi, left_center, left_radius1, (255, 255, 255), -1)
                    cv2.circle(right_roi, right_center, right_radius1, (255, 255, 255), -1)
                    if i != 1:
                        cv2.circle(left_roi, left_center, left_radius2, (0, 0, 0), -1)
                        cv2.circle(right_roi, right_center, right_radius2, (0, 0, 0), -1)
                    left_roi = np.int8(left_roi)
                    right_roi = np.int8(right_roi)

                    image1 = image.copy()
                    left_ring = cv2.bitwise_and(image1, image1, mask=left_roi)
                    image2 = image.copy()
                    right_ring = cv2.bitwise_and(image2, image2, mask=right_roi)
                    if i != 1:
                        left_area = np.pi * (left_radius1 * left_radius1 - left_radius2 * left_radius2)
                        _, left_G1, _ = cv2.split(left_ring)


                        right_area = np.pi * (right_radius1 * right_radius1 - right_radius2 * right_radius2)
                        _, right_G1, _ = cv2.split(right_ring)

                    else:
                        left_area = np.pi * (left_radius1 * left_radius1)
                        _, left_G2, _ = cv2.split(left_ring)


                        right_area = np.pi * (right_radius1 * right_radius1)
                        _, right_G2, _ = cv2.split(right_ring)

                    l = l + 1
                    if l <= 30:
                        left_avgpixels1.append(np.sum(left_G1) / left_area)
                        right_avgpixels1.append(np.sum(right_G1) / right_area)
                        left_avgpixels2.append(np.sum(left_G2) / left_area)
                        right_avgpixels2.append(np.sum(right_G2) / right_area)
                    else:
                        del (left_avgpixels1[0])
                        del (right_avgpixels1[0])
                        del (left_avgpixels2[0])
                        del (right_avgpixels2[0])
                        left_avgpixels1.append(np.sum(left_G1) / left_area)
                        right_avgpixels1.append(np.sum(right_G1) / right_area)
                        left_avgpixels2.append(np.sum(left_G2) / left_area)
                        right_avgpixels2.append(np.sum(right_G2) / right_area)

                if l%step_size==0:
                    raw_signals=[]
                    raw_signals.append(bandpass_filter(left_avgpixels1[:]))
                    raw_signals.append(bandpass_filter(right_avgpixels1[:]))
                    raw_signals.append(bandpass_filter(left_avgpixels2[:]))
                    raw_signals.append(bandpass_filter(right_avgpixels2[:]))
                    signals.append(raw_signals)
            else:
                break
        cap.release()
        all_signals.append(signals)
    np.savez('all_raw_signals',signals_save_path)


