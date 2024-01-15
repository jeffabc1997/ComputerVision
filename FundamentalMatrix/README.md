## Epipolar Geometry, Eight-point algorithm, Normalized eight-point algorithm, Fundamental matrix, Rank-2 Constraint, Homography, Image Rectification

# Epipolar Geometry, Eight-point algorithm, Normalized eight-point algorithm, Fundamental matrix, Rank-2 Constraint
## Fundamental Matrix Estimation from Point Correspondences: 
In this problem, you will implement both the linear least-squares version of the eight-point algorithm and its normalized version to estimate the fundamental matrices. You will implement the methods and complete the following:
- (a) Implement the linear least-squares eight-point algorithm and report the returned fundamental matrix.
Remember to enforce the rank-two constraint for the fundamental matrix via singular value decomposition. Briefly describe your implementation in your written report.
- (b) Implement the normalized eight-point algorithm and report the returned fundamental matrix. Remember to enforce the rank-two constraint for the fundamental matrix via singular value decomposition. Briefly describe your implementation in your written report.
- (c) Plot the epipolar lines for the given point correspondences determined by the fundamental matrices computed from (a) and (b). Determine the accuracy of the fundamental matrices by computing the average distance between the feature points and their corresponding epipolar lines.