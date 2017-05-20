# **Finding Lane Lines on the Road** 

## Final output

![alt text]['out.png']

 ### My pipeline consists of 5 main steps:
    1. Sanitize the image
        - Convert the image from RBG to grayscale
        - Apply Gaussian blurring
    2. Extract the edges using the canny method and mask the triangular region
        - For region masking, I did not want to simply hard-code the apex and left/right edge points. So, with some googling around and trial/error, I was able to generalize the region masking's apex, left/ and right coordinates
    3. Detect the lines using the Hough transform
        - I found that values like rho, theta, etc where quite elastic to change
    4. Extract the left and right lane markings
    5. Paint the lines over the original image
  
### Adjustments to draw_lines()
    I create a class called LaneFinder. This class has method called create_lines() which extrapolates left and right lines using their slopes.
    
    A positive slope signifies a left lane line, and a negative one signifies a right lane lane.

    I use the results of this method and pass it to draw_lines().  I wanted my pipeline to look clean and semantic, so I put all methods in this LaneFinder class, and chained one method call after the next.
    
### Shortcomings
    1. We assume that nothing is obstructing the lane lines. Snow, a person,vehcile or any other object could hide part or all of a lane line. 
    2. This pipeline would not work well for a road with a curve. That is, a road with edges not triangular in shape would not be captured as well by this pipeline

### Improvements
    1. Instead of looking for lines, we could look for a general space in which a lane line is most likely to be. This approach would be robust when the lane is blocked by snow, a vehicle or a person
    2. We could modify Hough transform to detect curved lines, too. This would require us to add more dimensions to the Hough space.
