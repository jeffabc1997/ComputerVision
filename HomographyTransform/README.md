# Homography, Image Rectification

## Homography transform: 
You need to determine a homograpgy transformation for plan-to-plane transformation. The homography transformation is determined by a set of point correspondences between the source image and the target image.
- (a) Implement a function that estimates the homography matrix H that maps a set of interest points to a new set of interest points. Describe your implementation.
- (b) Specify a set of point correspondences for the source image of the Delta building and the target one.
Compute the 3X3 homography matrix to rectify the front building of the Delta building image. The rectification is to make the new image plane parallel to the front building as best as possible. Please select four corresponding straight lines to compute the homograph matrix. Describe your implementation and show the selected correspondence line pairs, the homography matrix, and the rectified image. (Please use backward warping and bilinear interpolation)