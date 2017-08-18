# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:05:54 2017

@author: Yunshi_Zhao
"""

import pickle
import cv2
import numpy as np
import matplotlib.image as mpimg
import glob

def calibrate_cam():
    img_size = (1280, 720)
    nx, ny = 9, 6
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    objpoints = []
    imgpoints = []
    
    files = glob.glob('./camera_cal/calibration*.jpg')
    for idx, file in enumerate(files):
        img = mpimg.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray,
                                                 (nx, ny),
                                                  None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            #cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #outfile = './camera_cal/with_corners' + str(idx) + '.jpg'
            #mpimg.imsave(outfile, img) 
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                       imgpoints, img_size, None, None )
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump(dist_pickle, open('./camera_cal/dist_pickle.p', 'wb'))
    return dist_pickle

calibrate_cam()