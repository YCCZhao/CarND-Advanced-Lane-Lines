## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, I wrote a software pipeline [insert link] to identify the lane boundaries in a video, a detailed writeup of the project is in ####[insert link]. 

The Project
---

The steps of the pipeline are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Examples and output included in the repo are folowing:

* The images for camera calibration are stored in the folder called `camera_cal`.  
* Distorated images and distoration matrix are stored in the folder caled `output_camera_cal`.
* The images in `test_images` to test the pipeline.  
* Output from each stage of your pipeline are stored in the folder called `ouput_images`, a description is included in the writeup for the project.
* The pipeline is validated on the video called `project_video.mp4` and the `challenge_video.mp4` video.

[Insert output video here]


Future improvement:
More works need to be done for difficult road condition, like one in the `harder_challenge.mp4`.
