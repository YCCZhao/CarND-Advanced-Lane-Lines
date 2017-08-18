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


def undistortion(dist, mtx, img):   
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


def perspective_trans(img, unwarped=False, ori=[[566,490],[760,490],[220,700],[1070,700]], birdeye=[[200,100],[1000,100],[200,700],[1000,700]]):
    img_size = (1280, 720)
    src = np.float32(ori)
    dst = np.float32(birdeye)
    M = cv2.getPerspectiveTransform(src, dst)
    
    if unwarped:
        Minv = np.linalg.inv(M)
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
    def __init__(self, ym_per_pix=3/70, xm_per_pix=3.7/790, min_points=2000, smooth_factor=30):
        self.ym_per_pix = ym_per_pix 
        self.xm_per_pix = xm_per_pix
        self.min_points = min_points
        self.smooth_factor = smooth_factor
        self.current_fit = None
        self.current_fit_m = None
        self.current_x = None  
        self.current_y = None  
        self.recent_fitted = []
        self.recent_fitted_m = []
        self.good_fitted = []
        self.good_fitted_m = []
        self.best_fit = None
        self.best_fit_m = None
        self.z = None
        self.diff = np.array([0.0,0.0,0.0]) 
        self.radius_of_curvature = None
         
    def set_current_poly_fit(self, x, y):
        self.current_x = x
        self.current_y = y
        if len(self.current_x) >= self.min_points:
            self.current_fit = np.polyfit(self.current_y, self.current_x, 2)
            self.current_fit_m = np.polyfit(self.current_y*self.ym_per_pix, self.current_x*self.xm_per_pix, 2)
        else:
            self.current_fit = None
            self.current_fit_m = None
            
    def append_fit(self):
        try:
            self.diff = np.array(np.absolute(self.current_fit_m - self.best_fit_m)/self.best_fit_m)
        except:
            self.diff = None

        self.recent_fitted.append(self.current_fit)
        self.recent_fitted_m.append(self.current_fit_m)


    def set_best_fit(self):
        last_from = -1*self.smooth_factor
        self.good_fitted = [x for x in self.recent_fitted[last_from:] if x is not None]
        self.good_fitted_m = [x for x in self.recent_fitted_m[last_from:] if x is not None]
        self.best_fit = np.mean(np.array(self.good_fitted), axis=0)
        self.best_fit_m = np.mean(np.array(self.good_fitted_m), axis=0)
        
    def set_curvature(self, image_height=720):   
        self.radius_of_curvature = ((1 + (2*self.best_fit_m[0]*image_height*self.ym_per_pix + self.best_fit_m[1])**2)**1.5) / np.absolute(2*self.best_fit_m[0])
       

def slide_windows(x_base, nonzerox, nonzeroy, lane_inds=[], minpix=50, margin=100, img_size=(720,1200), nwindows=9):
    window_height = np.int(img_size[0]//nwindows)
    x_current = x_base
    for window in range(nwindows):
        win_y_low = img_size[0] - (window+1)*window_height
        win_y_high = img_size[0] - window*window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        lane_inds.append(good_inds)
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))
    return np.concatenate(lane_inds)

def lane_line_fit(binary_warped):  
   
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
       
    midpoint = np.int(binary_warped.shape[1]//2)
    margin = 100
    max_diff = np.array([2.0, 5.0, 1.0 ])
    
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
        right_lane_inds = slide_windows(rightx_base, nonzerox, nonzeroy, right_lane_inds)

    if (leftLine.best_fit is not None): 
        left_lane_inds = ((nonzerox > (leftLine.best_fit[0]*(nonzeroy**2) + leftLine.best_fit[1]*nonzeroy + leftLine.best_fit[2] - margin)) \
                          & (nonzerox < (leftLine.best_fit[0]*(nonzeroy**2) + leftLine.best_fit[1]*nonzeroy + leftLine.best_fit[2] + margin))) 
    else: 
        leftx_base = np.argmax(histogram[:midpoint])
        left_lane_inds = slide_windows(leftx_base, nonzerox, nonzeroy, left_lane_inds)
           

    # Fit a second order polynomial to each
    leftLine.set_current_poly_fit(nonzerox[left_lane_inds], nonzeroy[left_lane_inds])
    rightLine.set_current_poly_fit(nonzerox[right_lane_inds], nonzeroy[right_lane_inds] )

    #print('l', leftLine.diff, 'r', rightLine.diff)

    if leftLine.diff is not None and (leftLine.diff > max_diff).any():
        #print("l",leftLine.diff)
        leftLine.current_fit = None
    
    if rightLine.diff is not None and (rightLine.diff > max_diff).any():
        #print("r",rightLine.diff)
        rightLine.curent_fit = None

 
    if leftLine.current_fit is None and rightLine.current_fit is not None:
        leftLine.current_fit = np.copy(rightLine.current_fit)
        leftLine.current_fit_m = np.copy(rightLine.current_fit_m)
        leftLine.current_fit[2] -= 650
        leftLine.current_fit_m[2] -= 3.7
        
    if rightLine.current_fit is None and leftLine.current_fit is not None:
        rightLine.current_fit = np.copy(leftLine.current_fit)
        rightLine.current_fit_m = np.copy(leftLine.current_fit_m)
        rightLine.current_fit[2] += 650
        rightLine.current_fit_m[2] += 3.7
    

    leftLine.append_fit()
    rightLine.append_fit()

    leftLine.set_best_fit()
    rightLine.set_best_fit()      
    
    leftLine.set_curvature()
    rightLine.set_curvature()
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = leftLine.best_fit[0]*ploty**2 + leftLine.best_fit[1]*ploty + leftLine.best_fit[2]
    right_fitx = rightLine.best_fit[0]*ploty**2 + rightLine.best_fit[1]*ploty + rightLine.best_fit[2]
    


    if (leftLine.radius_of_curvature/rightLine.radius_of_curvature > 20 or leftLine.radius_of_curvature/rightLine.radius_of_curvature < 1/20) and len(leftLine.recent_fitted) > 30:
        #print(leftLine.radius_of_curvature, 'm', rightLine.radius_of_curvature, 'm')    
        if leftLine.radius_of_curvature < 500:
            del leftLine.recent_fitted[-1]
            del leftLine.recent_fitted_m[-1]
            leftLine.set_best_fit()
        if rightLine.radius_of_curvature < 500:
            del rightLine.recent_fitted[-1]
            del rightLine.recent_fitted_m[-1]
            rightLine.set_best_fit()  

        
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    return color_warp

rightLine = Line()
leftLine = Line()


dist_pickle = pickle.load(open('./camera_cal/dist_pickle.p', 'rb'))
dist = dist_pickle['dist']
mtx = dist_pickle['mtx'] 

def pipeline(img,dist=dist,mtx=mtx):  
    img = undistortion(dist, mtx, img)
    edges = edge_detection(img)
    binary_warped = perspective_trans(edges)
    color_warp = lane_line_fit(binary_warped)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = perspective_trans(color_warp, unwarped=True) 
    # Combine the result with the original image
    result = cv2.addWeighted(img[:,:,:3], 1, newwarp, 0.3, 0)
    return result
    
"""
files = glob.glob('../advanced_lane_finding/images/challenge/original/*.jpg')
for file in files:
    img = cv2.imread(file)
    res = pipeline(img)
    out = '../advanced_lane_finding/images/challenge/output/'+file.split('\\')[1].split('.')[0]+'.jpg'
    cv2.imwrite(out, res)
"""
from moviepy.editor import VideoFileClip

output = './output_video/challenge_video.mp4'
clip1 = VideoFileClip('./challenge_video.mp4')
output_clip = clip1.fl_image(pipeline) 
output_clip.write_videofile(output, audio=False) 


