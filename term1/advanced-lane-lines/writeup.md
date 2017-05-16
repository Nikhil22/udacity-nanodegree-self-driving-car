## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image1]: ./output_images/calibration5_undistorted.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/edges.png "Binary Example"
[image4]: ./output_images/beye1.png "Birds eye 1"
[image5]: ./output_images/beye2.png "Birds eye 2"
[image6]: ./output_images/final.png "Output"
[video1]: ./project_output_colour.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

1. Prepared object points. (x, y, z) coordinates of the chessboard corners, where z=0
2. object_pts appended with replicated array of coordinates everytime there is successful detection of all chessboard corners in a test image
3. img_points will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 
4. output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function
5. applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

1. Peform a camera calibration using cv2.calibrateCamera -> ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_pts, img_points, (img.shape[1], img.shape[0]),None,None)
2. call cv2.undistort using mtx, dist from step 1 -> cv2.undistort(orig, mtx, dist, None, mtx)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (code snippet below).  Here's an example of my output for this step. 

I used the HLS, LUV and Lab channels for this. I found that combining all 3 allowed me to better identify the yellow and white lane lines

Here's the code for this :
```python
def apply_binary_thresholds(img, thresholds={  \
      's': {'min': 180, 'max': 255}, \
      'l': {'min': 255, 'max': 255},   \
      'b': {'min': 155, 'max': 200}  \
    } , should_display=True): 
    
    S = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]  
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,0]
    B = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:,:,2] 

    s_bin = np.zeros_like(S)
    s_bin[(S >= thresholds['s']['min']) & (S <= thresholds['s']['max'])] = 1
    b_bin = np.zeros_like(B)
    b_bin[(B >= thresholds['b']['min']) & (B <= thresholds['b']['max'])] = 1
    l_bin = np.zeros_like(L)
    l_bin[(L >= thresholds['l']['min']) & (L <= thresholds['l']['max'])] = 1
    
    full_bin = np.zeros_like(s_bin)
    full_bin[(l_bin == 1) | (b_bin == 1) | (s_bin == 1)] = 1

    if should_display is True:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
        f.tight_layout()
        ax1.set_title('original image', fontsize=16)
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8'))
        ax2.set_title('all thresholds', fontsize=16)
        ax2.imshow(full_bin, cmap='gray')
        
    return full_bin
```
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I use the cv2.getPerspectiveTransform function.  My src and dst were made from 2 variables I declared:
```python
SRC_PTS = [[490, 482],[810, 482],[1250, 720], [40, 720]]
DST_PTS = [[0, 0], [1280, 0],[1250, 720], [40, 720]]
```
Here's the full code snippet
```python
def apply_birds_eye(img, should_display=True):
    img_shape = (img.shape[1], img.shape[0])
    
    src = np.float32(SRC_PTS)
    dst = np.float32(DST_PTS)
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_shape)
    
    if should_display is True:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Undistorted Image', fontsize=20)
        ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        ax2.set_title('Warped Image', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    return warped, M, Minv
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 490, 482      | 0, 0          | 
| 810, 482      | 1280, 0       |
| 1250, 620     | 1250, 720     |
| 40, 720       | 40, 720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I created a histogram of the image, found the left and right peaks and accordingly appended my left and right cooridnate arrays.

Here's a code snippet.
```python
hist = np.sum( \
            img[ \
                int( \
                     ry \
                ):int( \
                    ly \
                ), int(x_offset):int(width - x_offset) \
            ], axis=0 \
        )

        smoothened = signal.medfilt(hist, kernal_size)

        lt = np.array(signal.find_peaks_cwt(smoothened[:half], np.arange(1, 10)))
        rt = np.array(signal.find_peaks_cwt(smoothened[half:], np.arange(1, 10)))
        
        print("lt", lt)
        print("rt", rt)

        if len(lt) > 0:
            lxm.append(max(lt))

        if len(rt) > 0:
            rxm.append(max(rt) + half)

        if len(lt) > 0 or len(rt) > 0:
            ym.append((ly + ry) // 2)
            
        for lx_centre, centre_y in zip(lxm, ym):
            left_x_additional, left_y_additional = get_pxs(img, lx_centre,
                                                                       centre_y, g_rad // 2)

            lx_arr.append(left_x_additional)
            ly_arr.append(left_y_additional)
    
        for rx_centre, centre_y in zip(rxm, ym):
            right_x_additional, right_y_additional = get_pxs(img, rx_centre,
                                                                         centre_y, g_rad // 2)
            
            rx_arr.append(right_x_additional)
            ry_arr.append(right_y_additional)
```
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

```python
def pos_cen(y, left_poly, right_poly):
    center = (1.5 * polynomial_lines(y, left_poly)
              - polynomial_lines(y, right_poly)) / 2
    return center

lc_radius = np.absolute(((1 + (2 * lcs[0] * 500 + lcs[1])**2) ** 1.5) \
                /(2 * lcs[0]))
rc_radius = np.absolute(((1 + (2 * rcs[0] * 500 + rcs[1]) ** 2) ** 1.5) \
                 /(2 * rcs[0]))

ll_img = cv2.add( \
    cv2.warpPerspective( \
        painted_b_eye, Minv, (shape[1], shape[0]), flags=cv2.INTER_LINEAR \
    ), undistorted \
) 
plt.imshow(ll_img)
annotate(ll_img, curvature=(lc_radius + rc_radius) / 2, 
                     pos=pos_cen(719, lcs, rcs), 
                     curve_min=min(lc_radius, rc_radius))
plt.imshow(ll_img)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_output_colour.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had a little trouble with keeping the highlighted lane area exactly over the lane lines at first. 
My pipeline may fail in snowy conditions where lane lines are blocked. The edge detection algorithm would not be able to find lines.
An improvement could be to look for a general area of where lane lines are most likely to be located, rather than explictly looking for them.

I tried my pipeline for the challenge video. For the most part the lane area was highlighted accurately, but there was quite a bit of flickering in the highlighting. 
