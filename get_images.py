# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 20:29:26 2017

@author: Yunshi_Zhao
"""

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
vidcap = cv2.VideoCapture('./challenge_video.mp4')
framerate = int(vidcap.get(cv2.CAP_PROP_FPS))
framecount = 0

while vidcap.isOpened():
    success, image = vidcap.read()
    if success:
        #if framecount % framerate == 0:
        cv2.imwrite('../advanced_lane_finding/images/challenge/original/output_'+str(framecount).zfill(3)+'.jpg', image)
        framecount += 1
    else:
        break
cv2.destroyAllWindows()
vidcap.release()
"""


region1 = np.zeros((450, 1280), dtype=np.int8)
region2 = np.ones((250, 1280), dtype=np.int8)
region3 = np.zeros((20, 1280), dtype=np.int8)
area_interest = np.concatenate((region1,region2,region3), axis=0)

gray_threshold = 200
l_threshold = 200
s_thresh_min = 100
s_thresh_max = 255
yellow_thresh_min = 22
yellow_thresh_max = 25


files = glob.glob('./test_images/test_dst*.jpg')
for idx, file in enumerate(files):
    img = cv2.imread(file)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h,l,s = hls[:,:,0],hls[:,:,1],hls[:,:,2]
    
    l_binary = np.zeros_like(l)
    l_binary[(l > l_threshold)] = 1
    
    s_binary = np.zeros_like(s)
    s_binary[(s >= s_thresh_min) & (s <= s_thresh_max)] = 1
     
    combined1 = np.zeros_like(gray)
    combined1[ ((l_binary == 1) |(s_binary == 1))] = 1
    
    ksize = 9
    gradx = abs_sobel_thresh(s, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grady = abs_sobel_thresh(s, orient='y', sobel_kernel=ksize, thresh=(10, 255))  
    mag_binary = mag_thresh(s, sobel_kernel=ksize, mag_thresh=(20, 255))
    dir_binary = dir_threshold(s, sobel_kernel=ksize, thresh=(0.7, 1.3))
    
    combined2 = np.zeros_like(s)
    combined2[(((mag_binary == 1)&(dir_binary == 1))|((gradx == 1)&(grady == 1)))] = 1
    
    combined = np.zeros_like(s)
    combined[((combined1==1)|(combined2==1))&(area_interest == 1)] = 255
      
    outfile = './test_images/test_dst_gray_'+str(idx)+'.jpg'
    cv2.imwrite(outfile,combined)
 """