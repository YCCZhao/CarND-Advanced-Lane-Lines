# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 08:36:50 2017

@author: Yunshi_Zhao
"""


import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
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


dist_pickle = pickle.load(open('./camera_cal/dist_pickle.p', 'rb'))
dist = dist_pickle['dist']
mtx = dist_pickle['mtx'] 

def undistortion(dist, mtx, img):   
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def perspective_trans(img, unwarped=False):
    img_size = (1280, 720)
    src = np.float32([[566,490],[760,490],[220,700],[1070,700]])
    dst = np.float32([[200,100],[1000,100],[200,700],[1000,700]])
    M = cv2.getPerspectiveTransform(src, dst)
    
    if unwarped:
        Minv = np.linalg.inv(M)
        cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        transform = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    else:
        transform = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return transform


def abs_sobel_thresh(img_single_ch, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
    if orient == 'x':
        sobel = cv2.Sobel(img_single_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(img_single_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        raise ValueError('input x or y')
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return binary_output

def mag_thresh(img_single_ch, sobel_kernel=3, mag_thresh=(0, 255)):
    
    sobelx = cv2.Sobel(img_single_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_single_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelxy = (sobelx**2+sobely**2)**0.5
    scaled_sobelxy = np.uint8(255*sobelxy/np.max(sobelxy))
    binary_sobel = np.zeros_like(scaled_sobelxy)
    binary_sobel[(scaled_sobelxy < mag_thresh[1]) & (scaled_sobelxy > mag_thresh[0])] = 1
    return binary_sobel

def dir_threshold(img_single_ch, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    sobelx = cv2.Sobel(img_single_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_single_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel_angle = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(sobel_angle)
    binary_output[(sobel_angle < thresh[1]) & (sobel_angle > thresh[0])] = 1
    return binary_output


def edge_detection(img):
    
    region1 = np.zeros((450, 1280), dtype=np.int8)
    region2 = np.ones((250, 1280), dtype=np.int8)
    region3 = np.zeros((20, 1280), dtype=np.int8)
    area_interest = np.concatenate((region1,region2,region3), axis=0)
    
    l_threshold = 200
    s_thresh_min = 100
    s_thresh_max = 255
    
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h,l,s = hls[:,:,0],hls[:,:,1],hls[:,:,2]

    l_binary = np.zeros_like(l)
    l_binary[(l > l_threshold)] = 1
    
    s_binary = np.zeros_like(s)
    s_binary[(s >= s_thresh_min) & (s <= s_thresh_max)] = 1
     
    combined1 = np.zeros_like(s)
    combined1[((l_binary == 1) |(s_binary == 1))] = 1
    
    ksize = 9
    gradx = abs_sobel_thresh(s, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grady = abs_sobel_thresh(s, orient='y', sobel_kernel=ksize, thresh=(10, 255))  
    mag_binary = mag_thresh(s, sobel_kernel=ksize, mag_thresh=(20, 255))
    dir_binary = dir_threshold(s, sobel_kernel=ksize, thresh=(0.7, 1.3))
    
    combined2 = np.zeros_like(s)
    combined2[(((mag_binary == 1)&(dir_binary == 1))|((gradx == 1)&(grady == 1)))] = 1
    
    combined = np.zeros_like(s)
    combined[((combined1==1)|(combined2==1))&(area_interest == 1)] = 1

    return combined

class Line():
    def __init__(self):
        self.min_points = 1500
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        self.current_x = None  
        self.current_y = None  
        #difference in fit coefficients between last and new fits
        #Last n fits of the line
        self.recent_fitted = []
        self.good_fitted = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        self.std = None
        self.diff = None
        # was the line detected in the last iteration?
        self.detected = False  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
         
    def set_current_poly_fit(self):
        if len(self.current_x) >= self.min_points:
            self.current_fit = np.polyfit(self.current_y, self.current_x, 2)   
        else:
            self.current_fit = None   
            
    def append_fit(self):
        try:
            self.diff = ((self.current_fit - self.best_fit)/self.best_fit)#/self.std
        except:
            self.diff = None
        #if self.diff is not None and len(self.recent_fitted) > 10 and (abs(self.diff[0]) > 20 and abs(self.diff[1]) > 5 and abs(self.diff[2]) > 0.2):
        #    self.current_fit = None
        self.recent_fitted.append(self.current_fit)
        #else:
            #self.recent_fitted.append(None)
    
    def set_best_fit(self, last_from=30):
        last_from = -1*last_from
        temp = [x for x in self.recent_fitted[last_from:] if x is not None]
        self.best_fit = np.mean(np.array(temp), axis=0)
        self.std = np.std(np.array(temp), axis=0) / self.best_fit


            
def lane_mask(img,dist=dist,mtx=mtx):
    def slide_windows(x_base, lane_inds=[]):
        x_current = x_base
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))
        return np.concatenate(lane_inds)
    
    img = undistortion(dist, mtx, img)
    edges = edge_detection(img)
    #plt.imshow(edges,cmap='gray')
    binary_warped = perspective_trans(edges)
    #plt.imshow(binary_warped,cmap='gray')
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
       
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    midpoint = np.int(img.shape[0]//2)
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50    # Identify the x and y positions of all nonzero pixels in the image


    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    if (rightLine.best_fit is None) or (leftLine.best_fit is None):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    if (rightLine.best_fit is not None):
        right_lane_inds = ((nonzerox > (rightLine.best_fit[0]*(nonzeroy**2) + rightLine.best_fit[1]*nonzeroy + rightLine.best_fit[2] - margin)) \
                           & (nonzerox < (rightLine.best_fit[0]*(nonzeroy**2) + rightLine.best_fit[1]*nonzeroy + rightLine.best_fit[2] + margin)))  
    
    else:
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        right_lane_inds = slide_windows(rightx_base, right_lane_inds)

    if (leftLine.best_fit is not None): 
        left_lane_inds = ((nonzerox > (leftLine.best_fit[0]*(nonzeroy**2) + leftLine.best_fit[1]*nonzeroy + leftLine.best_fit[2] - margin)) \
                          & (nonzerox < (leftLine.best_fit[0]*(nonzeroy**2) + leftLine.best_fit[1]*nonzeroy + leftLine.best_fit[2] + margin))) 
    else: 
        leftx_base = np.argmax(histogram[:midpoint])
        left_lane_inds = slide_windows(leftx_base, left_lane_inds)
           
    # Extract left and right line pixel positions
    leftLine.current_x = nonzerox[left_lane_inds]
    leftLine.current_y = nonzeroy[left_lane_inds] 
    rightLine.current_x = nonzerox[right_lane_inds]
    rightLine.current_y = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    leftLine.set_current_poly_fit()
    rightLine.set_current_poly_fit()



    if leftLine.current_fit is None and rightLine.current_fit is not None:
        leftLine.current_fit = np.copy(rightLine.current_fit)
        leftLine.current_fit[2] -= 700
    if rightLine.current_fit is None and leftLine.current_fit is not None:
        rightLine.current_fit = np.copy(leftLine.current_fit)
        rightLine.current_fit[2] += 700
        
    leftLine.append_fit()
    rightLine.append_fit()
    leftLine.set_best_fit(30)
    rightLine.set_best_fit(30)       
    
    
       
    #print(leftLine.best_fit)
    #print('fit',rightLine.best_fit)
    print('diff',rightLine.diff)
    print('diff',leftLine.diff)
    #print(rightLine.best_fit[2]-leftLine.best_fit[2])
    #print('left', leftLine.std)
    #print('std',rightLine.std)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    y_eval = np.max(ploty)   
    left_curverad = ((1 + (2*leftLine.best_fit[0]*y_eval + leftLine.best_fit[1])**2)**1.5) / np.absolute(2*leftLine.best_fit[0])
    right_curverad = ((1 + (2*rightLine.best_fit[0]*y_eval + rightLine.best_fit[1])**2)**1.5) / np.absolute(2*rightLine.best_fit[0])
    
    if left_curverad/right_curverad > 3 or left_curverad/right_curverad < 1/3:
        if leftLine.std[2] > 0.1:
            del leftLine.recent_fitted[-10:]
            leftLine.set_best_fit
        if rightLine.std[2] > 0.1:
            del rightLine.recent_fitted[-10:]
            rightLine.set_best_fit
            
    
            
    
    left_fitx = leftLine.best_fit[0]*ploty**2 + leftLine.best_fit[1]*ploty + leftLine.best_fit[2]
    right_fitx = rightLine.best_fit[0]*ploty**2 + rightLine.best_fit[1]*ploty + rightLine.best_fit[2]

    #print(right_fitx[720//2]-left_fitx[720//2])
    
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = perspective_trans(color_warp, unwarped=True) 
    # Combine the result with the original image
    result = cv2.addWeighted(img[:,:,:3], 1, newwarp, 0.3, 0)
    return result

rightLine = Line()
leftLine = Line()

"""
files = glob.glob('../advanced_lane_finding/images/challenge/original/*.jpg')
for file in files:
    img = cv2.imread(file)
    res = lane_mask(img)
    out = '../advanced_lane_finding/images/challenge/output/'+file.split('\\')[1].split('.')[0]+'.jpg'
    cv2.imwrite(out, res)
"""
from moviepy.editor import VideoFileClip

output = './output_video/project_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip('./challenge_video.mp4').subclip(0,5)
clip1 = VideoFileClip('./project_video.mp4')
white_clip = clip1.fl_image(lane_mask) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False) 

