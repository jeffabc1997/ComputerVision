# Gaussian Blur, Sobel Operator, Harris Corner Detection


## Corner Detection:

- a. Gaussian Smooth: Show the results of Gaussian smoothing for 𝜎=5 and kernel size=5 and 10 respectively.
- b. Intensity Gradient (Sobel edge detection): Apply the Sobel filters to the blurred images and compute the magnitude (2 images) and direction (2 images) of gradient. (You should eliminate weak gradients by proper threshold.)
- c. Structure Tensor: Use the Sobel gradient magnitude (with Gaussian kernel size=10) above to compute the structure tensor 𝐻 of each pixel. Show the images of the smaller eigenvalue of 𝐻 with window size 3x3 and 5x5.
- d. Non-maximal Suppression: Perform non-maximal suppression on the results above along with appropriate thresholding for corner detection.
