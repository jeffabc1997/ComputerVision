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

def interpolation(original_img, transformed_x, transformed_y): 
    
    map_i, map_j = int(transformed_x), int(transformed_y) # transformed point is in the middle of 4 points
    a = transformed_x - map_i # distance between point(x, y) and (i, j)
    b = transformed_y - map_j
    # bilinear
    pixel_rgb = np.zeros((3,))
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

def main():
    building  = cv2.imread("Delta-Building.jpeg")
    # plot line on the selected image
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot([[431,417,886,895],[417,895,431,886]],[[341, 801, 11, 998],[801, 998, 341, 11]],linewidth=3, color='red')
    ax.imshow(np.array(building).astype('uint8')) 
    #plt.show()
    plt.savefig("./output/selected_img.jpeg", dpi = 300)
    # find homography matrix to warp
    height_x, width_y = 1300, 1300
    output_place = np.zeros((height_x,width_y,3))
    building_corner = np.array([[341,431],[801,417], [11,886],[998, 895]]) # be careful the coordinate to map
    output_corner = np.array([[0, 0], [height_x, 0], [0, width_y], [height_x, width_y]])
    homography_matrix = find_homography_matrix(output_corner, building_corner)
    print("HW2-2 Homography Matrix:\n", homography_matrix)
    output = backward_warpping(building, output_place, homography_matrix)
    
    cv2.imwrite('./output/rectified_img.png', output) # rectified image

if __name__ == '__main__':
    main()
