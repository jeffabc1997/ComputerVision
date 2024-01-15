import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
def see_coordinate(filename):
    img  = cv2.imread(filename)
    # plot line on the selected image
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    #ax.plot([[431,417,886,895],[417,895,431,886]],[[341, 801, 11, 998],[801, 998, 341, 11]],linewidth=3, color='red')
    ax.imshow(np.array(img).astype('uint8')) 
    # plt.show()

def read_image(path):
    img = cv2.imread(path)
    img_gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

def SIFT(img, nfeat):
    
    siftDetector= cv2.SIFT_create(nfeatures = nfeat)  # depends on OpenCV version; nfeatures = 5000

    kp, descriptor = siftDetector.detectAndCompute(img, None)
    return kp, descriptor

# Part A: SIFT Matches
# calculate Euclidean distance
def euclidean_dis(arr1, arr2):

    diff = arr1 - arr2
    diff_square = np.square(diff)
    dist_Sumofsquare = np.sum(diff_square)
    return dist_Sumofsquare # return square, not square root

# find closest 2 pairs of descriptors
def matcher(kp1, descriptor1, kp2, descriptor2):
    # Euclidean Distance: find the 2 smallest pair
    kp1_len = len(kp1)
    kp2_len = len(kp2)
    
    matches_list = []
    dist_smallest = 10000000000
    dist_small_2nd = 10000000001 # a large number
    for x in range(kp1_len):

        dmatch_smallest = cv2.DMatch(0, 0, dist_smallest)
        dmatch_small_2nd = cv2.DMatch(0, 0, dist_small_2nd)
        
        for y in range(kp2_len):
            dist = euclidean_dis(descriptor1[x], descriptor2[y])
            if dist <= dmatch_smallest.distance:
                
                dmatch_small_2nd = dmatch_smallest
                dmatch_smallest = cv2.DMatch(x,y, dist)
            else:
                if dist < dmatch_smallest.distance:
                    dmatch_small_2nd = cv2.DMatch(x, y, dist)
        
        dmatch_smallest.distance = np.sqrt(dmatch_smallest.distance) # square root the sum of square
        dmatch_small_2nd.distance = np.sqrt(dmatch_small_2nd.distance)
        
        matches_list.append(tuple((dmatch_smallest, dmatch_small_2nd)))

    matches = tuple(matches_list)
    return matches # return pair that descriptors from both images is close with index for descriptor left and right

# Apply ratio test
def ratio_test(matches, threshold, kp1, kp2):
    
    good = []
    for m,n in matches:
        if m.distance < threshold * n.distance:
            good.append([m])

    matches = []
    for pair in good:
        matches.append(list(kp1[pair[0].queryIdx].pt + kp2[pair[0].trainIdx].pt))

    matches = np.array(matches)
    return matches

# Save images, Plot matching points with lines
def plot_matches(matches, total_img, filename):
    plt.rcParams['figure.figsize'] = [15, 15] 
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr', markersize = 5)
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr', markersize = 5)
    
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
             linewidth=0.5)
    # plt.savefig(f"./output/{filename}.jpg", dpi = 300)
    #plt.show()

def sift_img_concatenate(book_image, image_total, interest_num, threshold):
    left_gray, left_origin, left_rgb = read_image(book_image)
    right_gray, right_origin, right_rgb = read_image(image_total)

    kp_left, descriptor_left = SIFT(left_gray, interest_num)
    kp_right, descriptor_right = SIFT(right_gray, interest_num)
    
    close_matches = matcher(kp_left, descriptor_left, kp_right, descriptor_right)

    matches_threshold = ratio_test(close_matches, threshold, kp_left, kp_right)

    blank_image = np.zeros(right_origin.shape, np.uint8)
    
    height, width = left_gray.shape[0], left_gray.shape[1]
    left_extend = blank_image.copy()
    left_extend[0:height, 0:width] = left_rgb.copy()
    total_img = np.concatenate((left_extend, right_rgb), axis=1)
    
    return matches_threshold, total_img

# Part B: RANSAC
def find_homography_matrix(src, des):
    # 4 points to make 8 rows
    A = np.array([[src[0][0], src[0][1], 1, 0, 0, 0, -1 * src[0][0] * des[0][0], -1 * src[0][1] * des[0][0], -1*des[0][0]],
                  [0, 0, 0, src[0][0], src[0][1], 1, -1 * src[0][0] * des[0][1], -1 * src[0][1] * des[0][1], -1*des[0][1]],
                  [src[1][0], src[1][1], 1, 0, 0, 0, -1 * src[1][0] * des[1][0], -1 * src[1][1] * des[1][0], -1*des[1][0]],
                  [0, 0, 0, src[1][0], src[1][1], 1, -1 * src[1][0] * des[1][1], -1 * src[1][1] * des[1][1], -1*des[1][1]],
                  [src[2][0], src[2][1], 1, 0, 0, 0, -1 * src[2][0] * des[2][0], -1 * src[2][1] * des[2][0], -1*des[2][0]],
                  [0, 0, 0, src[2][0], src[2][1], 1, -1 * src[2][0] * des[2][1], -1 * src[2][1] * des[2][1], -1*des[2][1]],
                  [src[3][0], src[3][1], 1, 0, 0, 0, -1 * src[3][0] * des[3][0], -1 * src[3][1] * des[3][0], -1*des[3][0]],
                  [0, 0, 0, src[3][0], src[3][1], 1, -1 * src[3][0] * des[3][1], -1 * src[3][1] * des[3][1], -1*des[3][1]]])
    
    U, D, Vh = np.linalg.svd(A) # least square solution
    homo_matrix = Vh[-1].reshape((3,3))
   
    return homo_matrix

def random_point(matches, k):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    point = np.array(point)
    src = point[:, :2] # x, y from left picture
    des = point[:, 2:] # x, y from right picture

    return src, des, point
def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1) # set homogeneous coordinate
    #print(all_p1)
    all_p2 = points[:, 2:4] # [:,2] is x,  [:, 3] is y
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.matmul(H, all_p1[i]) # p2' = Hp1, temp = (x, y, z)
        estimate_p2[i] = (temp/temp[2])[0:2] # x/z, y/z, z/z, and only take x, y to estimate_p2
    # Compute error of each transformed point
    errors = np.square(np.linalg.norm(all_p2 - estimate_p2 , axis=1))

    return errors, estimate_p2

def ransac(matches, threshold, iters):
    num_best_inliers = 0
    
    for i in range(iters):
        src, des, points = random_point(matches, 4)

        H = find_homography_matrix(src, des)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors, estimateP2 = get_error(matches, H)
        idx = np.where(errors < threshold)[0] # find
        inliers = matches[idx]
        estimate_inlier = estimateP2[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            estimate_inlier_best = estimate_inlier.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    #print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    #print("best_inlier: ", best_inliers)
    return best_inliers, best_H, estimate_inlier_best

def compute_corner(book_corner, homography_matrix):
    estimate_corner = np.zeros((4, 2))
    for i in range(4):
        temp = np.dot(homography_matrix, book_corner[i]) 
        estimate_corner[i] = (temp/temp[2])[0:2] # transformed 2D corner point
    return estimate_corner

def plot_deviation_vector(matches, estimate_points, estimate_corner ,total_img, filename):

    plt.rcParams['figure.figsize'] = [15, 15] 
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr', markersize = 5) # plot match (a, b) a points
    
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr', markersize = 5) # plot match (a, b) b points
    ax.plot(estimate_points[:, 0] + offset, matches[:, 3], 'oy', markersize = 3, alpha = 0.5) # plot b' points
    # plot matches' line
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]], linewidth=0.5) 
    # plot deviation vector line
    ax.plot([estimate_points[:, 0] + offset, matches[:, 2] + offset], [estimate_points[:, 1], matches[:, 3]], linewidth=0.5) 
    
    # draw transformed corner from original book
    estimate_corner[:, 0] = estimate_corner[:, 0] + offset
    for_draw_corner = np.zeros(estimate_corner.shape) # create different pairs for drawing lines
    for_draw_corner[3] = estimate_corner[0]
    for i in range(3):
        for_draw_corner[i] = estimate_corner[i+1]
    ax.plot([estimate_corner[:, 0], for_draw_corner[:, 0]],[estimate_corner[:, 1], for_draw_corner[:, 1]], 'g', linewidth = 4)

    plt.ylim([total_img.shape[0], 0]) # set y axis limit
    ax.plot()
    # plt.savefig(f"./output/{filename}.jpg", dpi = 300)
    #plt.show()

def main():
    

    book1 = "1-book1.jpg"
    book2 = "1-book2.jpg"
    book3 = "1-book3.jpg"  
    image_all = "1-image.jpg"

    # Get sift matches
    matches1_good, book1_and_total = sift_img_concatenate(book1, image_all, 2000, 0.45)
    matches2_good, book2_and_total = sift_img_concatenate(book2, image_all, 2000, 0.45)
    matches3_good, book3_and_total = sift_img_concatenate(book3, image_all, 8000, 0.6)
    output1_1aSift = "1a_book1_sift"
    output2_1aSift = "1a_book2_sift"
    output3_1aSift = "1a_book3_sift"
    # Plot matches
    plot_matches(matches1_good, book1_and_total, output1_1aSift)
    plot_matches(matches2_good, book2_and_total, output2_1aSift)
    plot_matches(matches3_good, book3_and_total, output3_1aSift)

    best_inliers1, best_H1, estimate_inlier_book1 = ransac(matches1_good, 3.5, 20)
    best_inliers2, best_H2, estimate_inlier_book2 = ransac(matches2_good, 3.5, 20)
    best_inliers3, best_H3, estimate_inlier_book3 = ransac(matches3_good, 3.2, 500)
 
    # see_coordinate(book1) # check corner coordinate
    # see_coordinate(book2) 
    # see_coordinate(book3) 
    # convert points to (x,y,z)
    book1_corner_original = cv2.convertPointsToHomogeneous(np.array([[17.5, 46.4], [11.2, 321.6], [433.8, 316.2], [422, 42.9]])).reshape((4,3)).astype(np.float32)
    book2_corner_original = cv2.convertPointsToHomogeneous(np.array([[28.4, 15.5], [24.4, 321.3], [427.1, 321.7], [408, 12]])).reshape((4,3)).astype(np.float32)
    book3_corner_original = cv2.convertPointsToHomogeneous(np.array([[18, 32], [18, 303], [424, 296], [416.3, 29.3]])).reshape((4,3)).astype(np.float32)
    # Use Homography matrix to compute corners for plotting
    book1_corner_homo = compute_corner(book1_corner_original, best_H1)
    book2_corner_homo = compute_corner(book2_corner_original, best_H2)
    book3_corner_homo = compute_corner(book3_corner_original, best_H3)   
    output1_3bDeviation = "1b_book1_deviation"
    output2_3bDeviation = "1b_book2_deviation"
    output3_3bDeviation = "1b_book3_deviation"
    # Plot corner lines, deviation vector, and better match lines
    plot_deviation_vector(best_inliers1, estimate_inlier_book1, book1_corner_homo, book1_and_total, output1_3bDeviation)
    plot_deviation_vector(best_inliers2, estimate_inlier_book2, book2_corner_homo, book2_and_total, output2_3bDeviation)
    plot_deviation_vector(best_inliers3, estimate_inlier_book3, book3_corner_homo, book3_and_total, output3_3bDeviation)
    

if __name__ == '__main__':
    main()
    
