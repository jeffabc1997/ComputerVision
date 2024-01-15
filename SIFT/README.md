# SIFT

## A. SIFT interest point detection
- a. Apply SIFT interest point detector (functions from OpenCV) to the following two images
- b. Adjust the related thresholds in SIFT detection such that there are around 100 interest points detected in each image .
- c. Plot the detected interest points on the corresponding images.

`SIFT(img, nfeat)` 使用`cv2.SIFT_create(nfeatures = nfeat)` 選出keypoint並且用nfeatures決定keypoint數量。Part A即可利用output出來的keypoint畫出來。

## B. SIFT feature matching
- a. Compare the similarity between all the pairs between the detected interest points from each of the two images based on a suitable distance function between two SIFT feature vectors.
- b. Implement a function that finds a list of interest point correspondences based on nearest-neighbor matching principle.
- c. Plot the point correspondences (from the previous step) overlaid on the pair of original images.

PartB用 `SIFT(img, nfeat)`中的`detectAndCompute()` 計算出keypoint的descriptor。
計算原理：對圖像做高斯模糊、對圖片降維，透過與下一個維度的圖片相減，再尋找上下相鄰維度的26個點中的local maxima...(略)，然後利用magnitude與orientation建立histogram...(略)，最後建立出一個128維的descriptor。
接著用`matcher(kp1, descriptor1, kp2, descriptor2)`將左圖的descriptor當作一個點，去向右圖的descriptor計算Euclidean Distance，我們先計算出兩個最短和次短的descriptor distance。
然後用 `ratio_test(matches, threshold, kp1, kp2)` 比對最短和次短的distance的差距大不大，如果差距很大，代表這個最短的distance是顯著的，也就是我們要的真正的對應點。
最後再用 `plot_matches(matches, total_img)`把兩張原圖和有matching的灰階圖疊出來。
