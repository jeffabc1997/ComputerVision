# K-means, K-means++, Mean-shift

## Image segmentation:

- A.  Apply K-means on the image (RGB color space) and try it with three different K values (your K should be > 3) and show the results (3 images). You should use 50 random initial guesses to select the best result based on the objective function for each K. Please discuss the difference between the results for different K’s.
- B.  Implement K-means++ to have better initial guess (3 images). Please discuss the difference between (A) and (B).
- C.  Implement the mean-shift algorithm to segment the same colors in the target image. Select appropriate parameters in the Uniform Kernel on the RGB color space to achieve optimal image segmentation (show the clustered result), and then show the pixel distributions in the R*G*B feature space before and after applying mean-shift (see Unit7 p.31). (3 images)
- D.  In addition, combine the color and spatial information into the kernel for mean shift segmentation and find the optimal parameters for the best segmentation result. (1 image)
