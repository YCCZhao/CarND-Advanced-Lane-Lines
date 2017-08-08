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

def undistortion(img):
    dist_pickle = pickle.load(open('./camera_cal/dist_pickle.p', 'rb'))
    dist = dist_pickle['dist']
    mtx = dist_pickle['mtx'] 
    
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
     
    combined1 = np.zeros_like(gray)
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


def lane_mask(img):
    edges = edge_detection(img)
    binary_warped = perspective_trans(edges)

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[2*binary_warped.shape[0]//3:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    #out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)        
    
    # Color in left and right line pixels
    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw on and an image to show the selection window
    #window_img = np.zeros_like(binary_warped)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    #left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    #left_line_pts = np.hstack((left_line_window1, left_line_window2))
    #right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    #right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
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

"""
files = glob.glob('./test_images/test_dst*.jpg')
for file in files:
    img = cv2.imread(file)
    res = lane_mask(img)
    out = './test_images/output_'+file.split('\\')[1].split('.')[0]+'.jpg'
    cv2.imwrite(out, res)

"""
from moviepy.editor import VideoFileClip

white_output = './output_video/project_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
## clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip('./project_video.mp4')
white_clip = clip1.fl_image(lane_mask) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False) 
