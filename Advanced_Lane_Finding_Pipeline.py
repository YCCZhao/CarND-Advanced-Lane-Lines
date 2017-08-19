# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 08:36:50 2017

@author: Yunshi_Zhao
"""

# import libraries
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import sys
from moviepy.editor import VideoFileClip


# This function undistort image. It takes distorted image as inputs,
# as well as dist and mtx matrix resulted from camera calibration.
# It returns undistored image. This function expected colored image.
def undistortion(dist, mtx, img):   
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


# This function performs a perspective transform. It transforms frontview
# to bird-eye view by using default paramemter.
def perspective_trans(img, unwarped=False, 
                      ori=[[595,450],[688,450],[245,700],[1080,700]], 
                      birdeye=[[350,100],[950,100],[350,720],[950,720]]):
    
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


# This function calculates the gradient either in x or y direction,
# then returns pixel location with gradient above thresholds.
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


# This function calculates the gradient both in x and y direction, 
# then it calcuates magnitude and returns pixel location with gradient magnitude 
# above thresholds.
def mag_thresh(img_single_ch, sobel_kernel=3, mag_thresh=(0, 255)):
    
    sobelx = cv2.Sobel(img_single_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_single_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobelxy = (sobelx**2+sobely**2)**0.5
    scaled_sobelxy = np.uint8(255*sobelxy/np.max(sobelxy))
    binary_sobel = np.zeros_like(scaled_sobelxy)
    binary_sobel[(scaled_sobelxy < mag_thresh[1]) & (scaled_sobelxy > mag_thresh[0])] = 1
    return binary_sobel


# This function calculates the gradient both in x and y direction, 
# then it calcuates direction and returns pixel location with gradient direction
# within thresholds.
def dir_threshold(img_single_ch, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    sobelx = cv2.Sobel(img_single_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_single_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel_angle = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(sobel_angle)
    binary_output[(sobel_angle < thresh[1]) & (sobel_angle > thresh[0])] = 1
    return binary_output


# Apply image processing techniques to identify pixel locations of the lane lines.
# It returns a numpy array with same shape as original images, with 1's indicating 
# presence of lane lines.
def edge_detection(img):  
    
    """
    region1 = np.zeros((450, 1280), dtype=np.int8)
    region2 = np.ones((250, 1280), dtype=np.int8)
    region3 = np.zeros((20, 1280), dtype=np.int8)
    area_interest = np.concatenate((region1,region2,region3), axis=0)
    """
    # thresholds values
    l_threshold = 100
    s_thresh_min = 100
    s_thresh_max = 255
    
    # transform image from BGR color space to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h,l,s = hls[:,:,0],hls[:,:,1],hls[:,:,2]

    # filtering out darker pixels. this filter aims to find bright white and yellow lines
    l_binary = np.zeros_like(l)
    l_binary[(l > l_threshold)] = 1
    s_binary = np.zeros_like(s)
    s_binary[(s >= s_thresh_min) & (s <= s_thresh_max)] = 1   
    combined1 = np.zeros_like(s)
    combined1[((l_binary == 1) | (s_binary == 1))] = 1
    
    # calculate pixel gradients in s layer of the HLS color space to find the edges of the image, 
    # locates lane line by using gradient thresholds.
    
    ksize = 9
    gradx = abs_sobel_thresh(s, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grady = abs_sobel_thresh(s, orient='y', sobel_kernel=ksize, thresh=(10, 255))  
    mag_binary = mag_thresh(s, sobel_kernel=ksize, mag_thresh=(20, 255))
    dir_binary = dir_threshold(s, sobel_kernel=ksize, thresh=(0.7, 1.3))   
    combined2 = np.zeros_like(s)
    combined2[(((mag_binary == 1)&(dir_binary == 1))|((gradx == 1)&(grady == 1)))] = 1
    
    # combine both filters
    combined = np.zeros_like(s)
    combined[((combined1==1)|(combined2==1))] = 1

    return combined


# class object to keep track of lane line properties.
class Line():
    
    def __init__(self, ym_per_pix=3/70, xm_per_pix=3.7/790, min_points=2000, smooth_factor=10):
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
    
    # take x and y location of the points and return a poly fit.
    def set_current_poly_fit(self, x, y):
        self.current_x = x
        self.current_y = y
        # if number of points is under specified minimum value, no line 
        # is fit to avoid resulting inaccurate lines.
        if len(self.current_x) >= self.min_points:
            self.current_fit = np.polyfit(self.current_y, self.current_x, 2)
            self.current_fit_m = np.polyfit(self.current_y*self.ym_per_pix, self.current_x*self.xm_per_pix, 2)
        else:
            self.current_fit = None
            self.current_fit_m = None
    
    # append fit of current frame to fit history
    def append_fit(self):
        # calcuate difference between fit of current frame and average fit of last n frames
        try:
            self.diff = np.array(np.absolute(self.current_fit_m - self.best_fit_m)/self.best_fit_m)
        except:
            self.diff = None

        self.recent_fitted.append(self.current_fit)
        self.recent_fitted_m.append(self.current_fit_m)

    # calculate average fit for the last n iteratioins, n can be changed by users.
    def set_best_fit(self):
        last_from = -1*self.smooth_factor
        self.good_fitted = [x for x in self.recent_fitted[last_from:] if x is not None]
        self.good_fitted_m = [x for x in self.recent_fitted_m[last_from:] if x is not None]
        self.best_fit = np.mean(np.array(self.good_fitted), axis=0)
        self.best_fit_m = np.mean(np.array(self.good_fitted_m), axis=0)
    
    # calcuate curvature based on average fit
    def set_curvature(self, image_height=720):   
        self.radius_of_curvature = ((1 + (2*self.best_fit_m[0]*image_height*self.ym_per_pix + self.best_fit_m[1])**2)**1.5) / np.absolute(2*self.best_fit_m[0])
       

# This function find lane line by identifying x location of maximum edge points, 
# then sliding search windows along y direction to collect line points.
# This function is called by lane_line_fit().
def slide_windows(x_base, nonzerox, nonzeroy, lane_inds=[], minpix=50, margin=100, img_size=(720,1200), nwindows=9):
    
    window_height = np.int(img_size[0]//nwindows)
    x_current = x_base
    for window in range(nwindows):
        win_y_low = img_size[0] - (window+1)*window_height
        win_y_high = img_size[0] - window*window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) \
                     & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        lane_inds.append(good_inds)
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))
    return np.concatenate(lane_inds)


# This function decides explicitly which pixels are part of the lines
# and which belong to the left line and which belong to the right line.
# Then it uses these sets of point to fit a Polynomial each side.
def lane_line_fit(binary_warped, margin=100, max_diff=np.array([10.0,10.0,10.0])):  
    
    # find the points associated with lane lines, and put x and y location in two arrays
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    #calcuate the midpoint location of the image
    midpoint = np.int(binary_warped.shape[1]//2)
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Find the points associated with the lane lines.
    # If no previous fit available, find the x location with most points,
    # then caling function slide_window to find all the points assocaited with lane lines.
    # If previous fit available, find lane lines points within the margin of previous fit.
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
    
    # if current fit appares to be a outliner, omit curret fit. 
    # maximum difference between current fit and average fit over last few iteration
    # can be specified as an input of this function
    if leftLine.diff is not None and (leftLine.diff > max_diff).any():
        #print("l",leftLine.diff)
        leftLine.current_fit = None
    
    if rightLine.diff is not None and (rightLine.diff > max_diff).any():
        #print("r",rightLine.diff)
        rightLine.curent_fit = None

    # if either side does not have a fit for current frame, use the fit of the other side
    # to derive current fit
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
    
    # append current fit to fit history
    leftLine.append_fit()
    rightLine.append_fit()
    
    # calculate average fit over last few iterations
    leftLine.set_best_fit()
    rightLine.set_best_fit()      
    
    # calculate curvature in 'm'
    leftLine.set_curvature()
    rightLine.set_curvature()
    curverad = (leftLine.radius_of_curvature+rightLine.radius_of_curvature)/2
    # if two curvature are very different, remove the side with a very small curvature.
    if (leftLine.radius_of_curvature/rightLine.radius_of_curvature > 10 or leftLine.radius_of_curvature/rightLine.radius_of_curvature < 1/20) and len(leftLine.recent_fitted) > 10:
        #print(leftLine.radius_of_curvature, 'm', rightLine.radius_of_curvature, 'm')    
        if leftLine.radius_of_curvature < 100:
            del leftLine.recent_fitted[-1]
            del leftLine.recent_fitted_m[-1]
            leftLine.set_best_fit()
        if rightLine.radius_of_curvature < 100:
            del rightLine.recent_fitted[-1]
            del rightLine.recent_fitted_m[-1]
            rightLine.set_best_fit()  

    # generate x and y location of points to use to draw the polynomial fits on the input image
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = leftLine.best_fit[0]*ploty**2 + leftLine.best_fit[1]*ploty + leftLine.best_fit[2]
    right_fitx = rightLine.best_fit[0]*ploty**2 + rightLine.best_fit[1]*ploty + rightLine.best_fit[2]
    
    # create an image to draw on and an image to show the selection window
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
     
    # calculate the offest of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])//2
    center_diff = (camera_center-color_warp.shape[1]//2)*leftLine.xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'
    
    # recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    return color_warp, curverad, center_diff, side_pos

# initiate class object for the left and the right lines
rightLine = Line()
leftLine = Line()

# load the dist and mtx matrix resulted from camera calibration, 
# which will be use to undistort images.
dist_pickle = pickle.load(open('./camera_cal/dist_pickle.p', 'rb'))
dist = dist_pickle['dist']
mtx = dist_pickle['mtx'] 

# This is the pipeline function which calls function to
# 1: undistort an image.
# 2: thresholding to generate a binary image where the lane lines are clearly visible.
# 3: tranform the binary image to a bird eye view image.
# 4: find points associated with lane line and perform a polynomial fit for each line
# 5: warp the blank back to original image space using inverse perspective matrix (Minv)
# 6: Combine the result with the original image
def pipeline(img,dist=dist,mtx=mtx):  
    img = undistortion(dist, mtx, img)
    edges = edge_detection(img)
    binary_warped = perspective_trans(edges)
    color_warp, curverad, center_diff, side_pos = lane_line_fit(binary_warped)
    newwarp = perspective_trans(color_warp, unwarped=True) 
    result = cv2.addWeighted(img[:,:,:3], 1, newwarp, 0.3, 0)
    cv2.putText(result, 'Raidus of Curvature = '+str(round(curverad,3))+'(m)',
                (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff,3)))+'(m) '+side_pos+' of center',
                (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    return result
 
    
def main(input_video='./project_video.mp4', output_video='./output_video/project_video.mp4'):  
    """
    following code can be used to test on individual picture.
    files = glob.glob('../advanced_lane_finding/images/challenge/original/*.jpg')
    for file in files:
        img = cv2.imread(file)
        res = pipeline(img)
        out = '../advanced_lane_finding/images/challenge/output/'+file.split('\\')[1].split('.')[0]+'.jpg'
        cv2.imwrite(out, res)
    """
    clip = VideoFileClip(input_video)
    output_clip = clip.fl_image(pipeline) 
    output_clip.write_videofile(output_video, audio=False) 


if __name__ == '__main__':
    arg = sys.argv[1:]
    if len(arg) == 3:
        main(arg[1], arg[2])
    else:
        main()



