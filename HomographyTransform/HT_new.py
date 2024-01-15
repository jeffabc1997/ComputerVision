import numpy as np
import matplotlib.pyplot as plt
import cv2

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

def interpolation(original_img, transformed_x, transformed_y): # bilinear
    map_i, map_j = int(transformed_x), int(transformed_y) # transformed point is in the middle of 4 points
    a = transformed_x - map_i # distance between point(x, y) and (i, j)
    b = transformed_y - map_j
    
    pixel_rgb = np.zeros((3,))

    x_limit, y_limit = original_img.shape[0] - 1, original_img.shape[1] - 1 
    # can't search for the pixels not existing on the original img
    if (map_i+1 <= x_limit) & (map_i >= 0) & (map_j+1 <= y_limit) & (map_j >= 0):   
        pixel_rgb = (1 - a) * (1 - b) * original_img[map_i][map_j] + a * (1 - b) * original_img[map_i+1,map_j] +\
                a * b * original_img[map_i+1,map_j+1] + (1 - a) * b * original_img[map_i,map_j+1]
    return pixel_rgb

def backward_warpping(original_img, output, homography_matrix):
    height_x, width_y, three = output.shape
    
    # height x 3 x width homogeneous index array 
    # [[[x0,x0,x0,...,x],[0,1,2,...,],[1,1,...,1]], [[x1,...],[0,1,...],[1...]], ... ]
    blank_output = np.zeros((height_x,3,width_y))
    number_sequence_y = np.arange(width_y)
    blank_output[:, 1] = number_sequence_y
    blank_output[:, 2, :] = 1
    
    for x in range(height_x):
        blank_output[x, 0, :] = x
        pos_on_original = np.matmul(homography_matrix, blank_output[x]) # transform coordinate
        original_x, original_y = pos_on_original[0] / pos_on_original[2], pos_on_original[1]/pos_on_original[2] # x' = x/z
        for y in range(width_y):    
            rgb_on_original = interpolation(original_img, original_x[y], original_y[y])
            output[x][y] = rgb_on_original

    return output

def plot_original_img(original_img):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot([[431,417,886,895],[417,895,431,886]],[[341, 801, 11, 998],[801, 998, 341, 11]],linewidth=3, color='red')
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    ax.imshow(np.array(original_rgb).astype('uint8')) 
    #plt.show()
    plt.savefig("./output/selected_img.jpeg", dpi = 300)

def plot_rectified_img(transformed_img):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    ax.plot([[200,200,1100,1100],[200,1100,1100,200]],[[200, 900, 900, 200],[900, 900, 200, 200]],linewidth=3, color='red')
    
    transformed_rgb = cv2.cvtColor(transformed_img.astype(np.float32), cv2.COLOR_BGR2RGB)
    ax.imshow(np.array(transformed_rgb/255)) 

    plt.savefig("./output/rectified_img.jpeg", dpi = 300)

def main():
    building  = cv2.imread("Delta-Building.jpeg")
    # plot line on the selected image
    plot_original_img(building)

    # find homography matrix to warp

    building_corner = np.array([[341,431],[801,417], [11,886],[998, 895]]) # be careful the coordinate to map
    
    area_projection= np.array([200, 900, 200, 1100]) # projection_top, projection_down, projection_left, projection_right
    output_corner = np.array([[area_projection[0], area_projection[2]],[area_projection[1],area_projection[2]],\
                                [area_projection[0],area_projection[3]],[area_projection[1],area_projection[3]]])
    homography_matrix = find_homography_matrix(output_corner, building_corner)
    print("HW2-2 Homography Matrix:\n", homography_matrix)
    
    height_x, width_y = 1068, 1600
    output_place = np.zeros((height_x,width_y,3))
    
    output = backward_warpping(building, output_place, homography_matrix)
    plot_rectified_img(output)

if __name__ == '__main__':
    main()
