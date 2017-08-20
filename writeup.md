**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/with_corners1.jpg "Undistorted"
[image2]: ./examples/undistorted_output.jpg "Road Transformed"
[image3]: ./examples/threshold_output.jpg "Binary Example"
[image8]: ./examples/perspective_input.jpg "Perspective Transform Test Input"
[image9]: ./examples/perspective_output.jpg "Perspective Transform Test Output"
[image4]: ./examples/bird-eye_output.jpg "Warp Example"
[image5]: ./examples/poly_fit_output.jpg "Fit Visual"
[image7]: ./examples/newwarp_output.jpg "Masking Area"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./output_video/project_video.mp4 "Project Output Video"
[video2]: ./output_video/challenge_video.mp4 "Challenge Output Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code is  called `camera_calibration.py`. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![][image1] 

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using `cv2.undistort` function and the camera matrix and distance coefficient obtained from camera calibration steps, image is undistorted. A distortion-corrected example image is shown below:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 95 through 135 in `Advanced_Lane_Finding_Pipeline.py`). I used color threshold to filter all pixels with lightness under 200 and saturation 100 (line 103-118). This is meant to keep the clear lane lines. I also used graient thresholds to find edges (line 123-129). I chose saturation layer of the image to calcuate gradient after testing on different layers.

Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_trans()`, which appears in lines 30 through 43 in the file `Advanced_Lane_Finding_Pipeline.py` (./Advanced_Lane_Finding_Pipeline.py). The `perspective_trans()` function takes as inputs an image (`img`), and has default values for source (`src`) and destination (`dst`) points.  This points works for current camera location, but since different camera orientation is used, they might need to be updated.

```python
def perspective_trans(img, unwarped=False, 
                      ori=[[566,490],[760,490],[220,700],[1070,700]], 
                      birdeye=[[200,100],[1000,100],[200,700],[1000,700]]):
    
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
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595,450      | 350,100      | 
| 688,450      | 950,100      |
| 245,700      | 350,720      |
| 1080,700     | 950,720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
Here is the test input and output.
![alt text][image8]
![alt text][image9]

The output of example image at the stage is shown below
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

So once I had the binary warped image of the example image, I found for the x locations where the lower half the image had the most non-zero pixels, one for the left half of the image, and one for the right half of the image. Then I appended all the non-zero pixels inside the search window at that location. The serach window moved up and updated x location as it moved up. And everytime non-zero pixels within window were appended to a list. If previous fits were available, search windows were started at the previous poly fit instead of the x location with most non-zero pixels.  By using points found (left and right), polynomial fit was performed using `np.poly` function. Once lines were fit, I calcuated the difference between current fit and the average fit over last few iterations, if difference were too big, current fit was removed and the result from previous frame was used. In the case that only one fit was found, the other fit were derived using the fit identified, since lane lines are fairly parallel. When both fits were found, their curvatures were compared. If they are too different, current fit will be removed and the result from previous frame was used. Finally, x and y points of the poly fit were generated for visualization.

The output image at this state looks like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Right after both lines were fit, the curvature function (line 206 to 207) of the Line Class was called in line 315 to 316 of `Advanced_Lane_Finding_Pipeline.py`
Calculation of position of the vehicle with respect to center was performed in line 338 through 342 of `Advanced_Lane_Finding_Pipeline.py`. I first found the camera center location, then calcuated the distance between bemera and the lane lines. And that's the position of the vehicle with repect to either side of the lane.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 376 through 383 in my code in `Advanced_Lane_Finding_Pipeline.py` in the function `pipeline()`.  Here is an example of my result on a test image:

![alt text][image7]
![alt text][image6]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project video result](./video_output/project_video.mp4)
Here's a [link to my challenge video result](./video_output/challenge_video.mp4)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The color and gradient threshold method worked very well when there was enough brightness of a image. However it wasn't doing a great job when the image is very dark - i.e. hardly any pixels associated with lane lines were found when under the shadow of a bridge. If threshold were set low so that lane lines can be found when dark, lots noise would be picked up when bright. Therefore current implementation is not very robust. If I had more time, I would probably look into image equaliztion or other method to maximize image contrast regarding the brightness.

Another issue is curvature varied a lot from frame to frame which still needs some investigation.
