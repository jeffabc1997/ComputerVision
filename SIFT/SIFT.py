import cv2
import matplotlib.pyplot as plt
import numpy as np

def read_image(path):
    img = cv2.imread(path)
    img_gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img, img_rgb

def SIFT(img, nfeat):
    
    siftDetector= cv2.SIFT_create(nfeatures = nfeat)  # depends on OpenCV version; nfeatures = 5000

    kp, descriptor = siftDetector.detectAndCompute(img, None)
    return kp, descriptor

# Part A
# Save images
def plot_keypoint(kp1, kp2, total_img):
    plt.rcParams['figure.figsize'] = [15, 15] 
    match_img = total_img.copy()
    offset = total_img.shape[1]//2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    kp1_x = [i.pt[0] for i in kp1]
    kp1_y = [i.pt[1] for i in kp1]
    kp2_x = [i.pt[0] + offset for i in kp2]
    kp2_y = [i.pt[1] for i in kp2]

    ax.plot(kp1_x , kp1_y, 'y.', markersize = 2)
    ax.plot(kp2_x , kp2_y, 'y.', markersize = 2)

    #plt.savefig(".//2/output/A_InterestPoint_feat1500.png")
    #plt.show()

# Part B
# calculate Euclidean distance
def euclidean_dis(arr1, arr2):

    diff = arr1 - arr2
    diff_square = np.square(diff)
    dist_Sumofsquare = np.sum(diff_square)
    #dist = np.sqrt(dist_Sumofsquare)
    return dist_Sumofsquare

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
def plot_matches(matches, total_img):
    plt.rcParams['figure.figsize'] = [15, 15] 
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr', markersize = 5)
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr', markersize = 5)
    ax.set_prop_cycle(color=['blue', 'yellow', 'cyan']) # set color of the lines
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
             linewidth=0.5)
    #plt.savefig(".//2/output/B_Feature_Matching_feat2000thres0_45.png")
    #plt.show()

## A. SIFT interest point detection
left_gray, left_origin, left_rgb = read_image('1a_notredame.jpg')
right_gray, right_origin, right_rgb = read_image('1b_notredame.jpg')

interest_kp_left, interest_descriptor_left = SIFT(left_gray, 1500)
interest_kp_right, interest_descriptor_right = SIFT(right_gray, 1500)

right_rgb_resize = cv2.resize(right_rgb, (1536,2048))

total_2pics = np.concatenate((left_rgb, right_rgb_resize), axis=1)
plot_keypoint(interest_kp_left,interest_kp_right, total_2pics)

## B. SIFT feature matching
# SIFT only can use gray
kp_left, descriptor_left = SIFT(left_gray, 2000)
kp_right, descriptor_right = SIFT(right_gray, 2000)
matches_1st = matcher(kp_left, descriptor_left, kp_right, descriptor_right)

threshold = 0.45
matches_2nd = ratio_test(matches_1st, threshold, kp_left, kp_right)


right_rgb = cv2.resize(right_rgb, (1536,2048)) # resize for concatenation

total_img = np.concatenate((left_rgb, right_rgb), axis=1)

plot_matches(matches_2nd, total_img) # Plot matches and save
