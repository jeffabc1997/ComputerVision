import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import scipy.signal
import os

### A.a.Gaussian Blur
# Opencv assume BGR
def gaussian_BGR_blur(gaussian_kernel_norm, image_read):

    conv2d = scipy.signal.convolve2d
    # get single color
    blue = image_read[:, :, 0]
    green = image_read[:, :, 1]
    red = image_read[:, :, 2]
    # convolve each color
    blur_blue = conv2d(blue, gaussian_kernel_norm)
    blur_green = conv2d(green, gaussian_kernel_norm)
    blur_red = conv2d(red, gaussian_kernel_norm)
    # stack to right shape
    gaussian_blur = np.stack([ blur_blue, blur_green, blur_red ], axis=2)

    return gaussian_blur

# n x n Gassian filter (5x5 or 10x10)
omega = 5

# kernel size = 5
x, y = np.mgrid[-2 : 3, -2 : 3] # -2 to -2 in matrix
gaussian_kernel_5 = 1/(2*np.pi* (omega ** 2)) * np.exp(-(x**2+y**2) / (2 * omega ** 2)) 

# kernel size = 10
step10 = np.linspace(-4.5, 4.5, 10) # -4.5, ..., -0.5, 0.5, ..., 4.5
x10, y10 = np.meshgrid(step10, step10)
gaussian_kernel_10 = 1/(2*np.pi* (omega ** 2)) * np.exp(-(x10**2+y10**2) / (2 * omega ** 2)) 

# Normalization of kernel
gaussian_norm_kernel_5 = gaussian_kernel_5 / gaussian_kernel_5.sum()
gaussian_norm_kernel_10 = gaussian_kernel_10 / gaussian_kernel_10.sum()

currentDir = os.getcwd()
chess_original_image = cv2.imread(os.path.join(currentDir, "chessboard-hw1.jpg"))
Notre_original_image = cv2.imread(os.path.join(currentDir, "1a_notredame.jpg"))

chess_gaussian_blur_5 = gaussian_BGR_blur(gaussian_norm_kernel_5, chess_original_image)
chess_gaussian_blur_10 = gaussian_BGR_blur(gaussian_norm_kernel_10, chess_original_image)
Notre_gaussian_blur_5 = gaussian_BGR_blur(gaussian_norm_kernel_5, Notre_original_image)
Notre_gaussian_blur_10 = gaussian_BGR_blur(gaussian_norm_kernel_10, Notre_original_image)

### A.b.Intensity Gradient
def sobel_magnitude_angle(image_read):
    kernel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) # Sobel operator
    kernel_y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    conv2d = scipy.signal.convolve2d
    image_read = image_read.astype(np.float32) # original is float64
    try:
        bw = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY) # to grayscale
    except Exception as e:
        bw = image_read
        pass
    sobel_x = conv2d(bw, kernel_x) # derivative horizontally
    sobel_y = conv2d(bw, kernel_y) # derivative vertically
    sobel_magnitude = np.zeros(sobel_x.shape)
    sobel_magnitude = abs(sobel_x) + abs(sobel_y)
    sobel_angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi # arctan2 return [-pi, pi]

    return (sobel_magnitude, sobel_angle)

# take magnitude and angle these 2 input
def MagAngle_to_HSV(sobel_Mag_Angle):
    sobel_magnitude, sobel_angle = sobel_Mag_Angle
    
    sobel_angle[sobel_angle < 0] += 360 # turn -179 degree to 181
    sobel_hue  = (sobel_angle / 360) * 255 # range [0,255]

    sobel_saturation = np.full((sobel_hue.shape), 1) # saturation range [0,1]
    sobel_value = sobel_magnitude / np.amax(sobel_magnitude) * 255 # value range [0, 255]
    sobel_hsv = np.stack([ sobel_hue, sobel_saturation, sobel_value ], axis = 2).astype(np.float32)
    sobel_hsv2bgr = cv2.cvtColor(sobel_hsv, cv2.COLOR_HSV2BGR)
    return sobel_hsv2bgr


# Calculate magnitude and gradient direction
chess_sobel_MagAngle_Gauss5 = sobel_magnitude_angle(chess_gaussian_blur_5)
chess_sobel_MagAngle_Gauss10 = sobel_magnitude_angle(chess_gaussian_blur_10)
Notre_sobel_MagAngle_Gauss5 = sobel_magnitude_angle(Notre_gaussian_blur_5)
Notre_sobel_MagAngle_Gauss10 = sobel_magnitude_angle(Notre_gaussian_blur_10)

# Make magnitude and direction into HSV color style
chess_angle_hsv_Gauss5 = MagAngle_to_HSV(chess_sobel_MagAngle_Gauss5)
chess_angle_hsv_Gauss10 = MagAngle_to_HSV(chess_sobel_MagAngle_Gauss10)
Notre_angle_hsv_Gauss5 = MagAngle_to_HSV(Notre_sobel_MagAngle_Gauss5)
Notre_angle_hsv_Gauss10 = MagAngle_to_HSV(Notre_sobel_MagAngle_Gauss10)

# Make Magnitude images 
chess_sobel_magnitude_Gauss5 = chess_sobel_MagAngle_Gauss5[0]
chess_sobel_magnitude_Gauss10 = chess_sobel_MagAngle_Gauss10[0]
Notre_sobel_magnitude_Gauss5 = Notre_sobel_MagAngle_Gauss5[0]
Notre_sobel_magnitude_Gauss10 = Notre_sobel_MagAngle_Gauss10[0]

### A.c Structure Tensor (with Gaussian kernel size=10)
# Turn image into grayscale with convolution by specified kernel
def sobel_derivative_xy(image_read):
    kernel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) # sobel operator
    kernel_y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    conv2d = scipy.signal.convolve2d
    image_read = image_read.astype(np.float32)  # original is float64 maybe
    try:
        bw = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY) # grayscale
    except Exception as e:
        bw = image_read
        pass
    derivative_x = conv2d(bw, kernel_x) # derivative horizontally
    derivative_y = conv2d(bw, kernel_y)
    sobel_gradient = np.stack([ derivative_x, derivative_y ], axis=2).astype(np.float32)
    return sobel_gradient

# To get the structure tensor of every pixel
def structure_tensor_make(sobel_gradient):
    sobel_x = sobel_gradient[:, :, 0]
    sobel_y = sobel_gradient[:, :, 1]
    IxIy = sobel_x * sobel_y
    IxSquare = np.square(sobel_x)
    IySquare = np.square(sobel_y)
    structure_tensor_image = np.stack([ IxSquare, IxIy, IxIy, IySquare ], axis = 2) # every pixel has a 4-dimension tensor aka 'H'

    return structure_tensor_image

# To get the SSD of every pixel
def SSD_make(structure_tensor, window_size):
    
    summation_kernel = np.full((window_size, window_size), 1) 
    conv2d = scipy.signal.convolve2d # we use convolution to do the summation of nxn window size
    IxSquare_summation = conv2d(structure_tensor[:,:,0], summation_kernel)
    IxIy_summation = conv2d(structure_tensor[:,:,1], summation_kernel)
    IySquare_summation = conv2d(structure_tensor[:,:,3], summation_kernel)
    SSD_image = np.stack([ IxSquare_summation, IxIy_summation, IxIy_summation, IySquare_summation ], axis = 2) # every pixel has a SSD tensor = E(u, v)
    return SSD_image

# smaller eigenvalue = det(H) / tr(H)
def little_eigenvalue(SSD, threshold):

    SSD_reshape = SSD.reshape(SSD.shape[:-1] + (2,2)) # reshape to (width, height, 2, 2) because we need to use numpy functions
    
    trace_H = np.trace(SSD_reshape,axis1=2, axis2=3) # axis is used to calculate trace of 2x2 matrices
    determinant_H = np.linalg.det(SSD_reshape) # calculate determinant
    
    eigenvalue_smaller = np.zeros(determinant_H.shape, dtype = np.float32)
    eigenvalue_smaller = np.divide(determinant_H, trace_H, out=np.zeros_like(determinant_H), where=trace_H!=0)
    eigenvalue_smaller[eigenvalue_smaller[:,:] < threshold] = 0
    return eigenvalue_smaller


# calculate derivative x, y from sobel gradient magnitude
chess_sobel_direction_xy_gaussian_10 = sobel_derivative_xy(chess_sobel_magnitude_Gauss10)  
# Threshold 28 seems good
threshold_of_Magnitude = 45
# Threshold function: pixel < threshold => 0, others remain the original value
ret, Notre_Magnitude_Thresh_size10 = cv2.threshold(Notre_sobel_magnitude_Gauss10, threshold_of_Magnitude, 255,cv2.THRESH_TOZERO) 
Notre_sobel_direction_xy_gaussian_10 = sobel_derivative_xy(Notre_Magnitude_Thresh_size10) 
# Turn derivative into structure tensor of every pixel
structure_tensor_chess_gaussian_10 = structure_tensor_make(chess_sobel_direction_xy_gaussian_10)
structure_tensor_Notre_gaussian_10 = structure_tensor_make(Notre_sobel_direction_xy_gaussian_10)
# Calcualte SSD of tensors by window size
SSD_chess_gaussian_10_3x3 = SSD_make(structure_tensor_chess_gaussian_10,  3)
SSD_chess_gaussian_10_5x5 = SSD_make(structure_tensor_chess_gaussian_10,  5)
SSD_Notre_gaussian_10_3x3 = SSD_make(structure_tensor_Notre_gaussian_10,  3)
SSD_Notre_gaussian_10_5x5 = SSD_make(structure_tensor_Notre_gaussian_10,  5) # maybe we should use other threshold for 5x5 instead of 28
# Obtain small eigenvalue to know if it is an edge/corner
thresh_for_chess = 0
threshold_for_Notre = 300
small_eigenvalue_chess_window_3x3 = little_eigenvalue(SSD_chess_gaussian_10_3x3, thresh_for_chess)
small_eigenvalue_chess_window_5x5 = little_eigenvalue(SSD_chess_gaussian_10_5x5, thresh_for_chess)
small_eigenvalue_Notre_window_3x3 = little_eigenvalue(SSD_Notre_gaussian_10_3x3, threshold_for_Notre)
small_eigenvalue_Notre_window_5x5 = little_eigenvalue(SSD_Notre_gaussian_10_5x5, threshold_for_Notre)

### A.d Corner Detection with Non-Maximum Suppression 
# Harris Response(R) = det(H) - k * (tr(H)^2)
def harris_response(SSD_structure_tensor, threshold):
    SSD_reshape = SSD_structure_tensor.reshape(SSD_structure_tensor.shape[:-1] + (2,2))
    trace_H = np.trace(SSD_reshape,axis1=2, axis2=3) # axis is used to calculate trace of 2x2 matrices
    determinant_H = np.linalg.det(SSD_reshape) # calculate determinant
    k = 0.05
    response = determinant_H - k * (trace_H**2)
    response[response[:,:] < threshold] = 0
    return response

def non_max_suppression(image_read, window_size):
    #print(np.max(image_read))
    offset = window_size - window_size // 2 - 1
    row_len, col_len = image_read.shape
    for i in range(offset,row_len-offset):
        for j in range(offset,col_len-offset):
            window = image_read[i-offset : i+offset, j-offset : j+offset]
            if image_read[i][j] < np.max(window):
                
                image_read[i][j] = 0
    #print(np.max(image_read))
    return image_read

def overlay_img(orignal_img, nms_img, w1, w2):
    orignal_img = orignal_img.astype(np.float32)
    
    empty_arr = np.zeros(nms_img.shape)
    turn_red_nms = np.stack([empty_arr, empty_arr, nms_img], axis = 2)
    turn_red_nms_dilate = cv2.dilate(turn_red_nms, None).astype(np.float32)
    
    nms_dilate_resize = cv2.resize(turn_red_nms_dilate, (orignal_img.shape[1], orignal_img.shape[0]))
    overlay_weighted = cv2.addWeighted(orignal_img, w1, nms_dilate_resize, w2, 0)

    return overlay_weighted

harris_response_Notre10_3x3 = harris_response(SSD_Notre_gaussian_10_3x3, 100000)
harris_response_Notre10_5x5 = harris_response(SSD_Notre_gaussian_10_5x5, 100000)

nms3_chess10_H3x3 = non_max_suppression(small_eigenvalue_chess_window_3x3, 3)
nms3_chess10_H3x3[nms3_chess10_H3x3[:,:] < np.max(nms3_chess10_H3x3) * 0.005] = 0
chess10_nms3_h3x3_final = overlay_img(chess_original_image, nms3_chess10_H3x3, 0.95 ,0.1)
#NMS3_H3x3.jpg'), chess10_nms3_h3x3_final)

nms5_chess10_H3x3 = non_max_suppression(small_eigenvalue_chess_window_3x3, 5)
nms5_chess10_H3x3[nms5_chess10_H3x3[:,:] < np.max(nms5_chess10_H3x3) * 0.005] = 0
chess10_nms5_h3x3_final = overlay_img(chess_original_image, nms5_chess10_H3x3, 0.95 ,0.1)
#NMS5_H3x3.jpg'), chess10_nms5_h3x3_final)

nms3_chess10_H5x5 = non_max_suppression(small_eigenvalue_chess_window_5x5, 3)
nms3_chess10_H5x5[nms3_chess10_H5x5[:,:] < np.max(nms3_chess10_H5x5) * 0.01] = 0
chess10_nms3_h5x5_final = overlay_img(chess_original_image, nms3_chess10_H5x5, 0.95 ,0.15)
#NMS3_H5x5.jpg'), chess10_nms3_h5x5_final)

nms5_chess10_H5x5 = non_max_suppression(small_eigenvalue_chess_window_5x5, 5)
nms5_chess10_H5x5[nms5_chess10_H5x5[:,:] < np.max(nms5_chess10_H5x5) * 0.01] = 0
chess10_nms5_h5x5_final = overlay_img(chess_original_image, nms5_chess10_H5x5, 0.95 ,0.15)
#NMS5_H5x5.jpg'), chess10_nms5_h5x5_final)

nms3_Notre10_H3x3 = non_max_suppression(harris_response_Notre10_3x3, 3)
nms3_Notre10_H3x3[nms3_Notre10_H3x3[:,:] < np.max(nms3_Notre10_H3x3) * 0.01] = 0
notre10_nms3_h3x3_final = overlay_img(Notre_original_image, nms3_Notre10_H3x3, 0.85 ,0.15)
#cv2.imwrite(os.path.join(currentDir,'','1','output', 'normal','d_Notre10_NMS3_H3x3.jpg'), notre10_nms3_h3x3_final)

nms5_Notre10_H3x3 = non_max_suppression(harris_response_Notre10_3x3, 5)
nms5_Notre10_H3x3[nms5_Notre10_H3x3[:,:] < np.max(nms5_Notre10_H3x3) * 0.01] = 0
notre10_nms5_h3x3_final = overlay_img(Notre_original_image, nms5_Notre10_H3x3, 0.85 ,0.15)
#cv2.imwrite(os.path.join(currentDir,'','1','output', 'normal','d_Notre10_NMS5_H3x3.jpg'), notre10_nms5_h3x3_final)

nms3_Notre10_H5x5 = non_max_suppression(harris_response_Notre10_5x5, 3)
nms3_Notre10_H5x5[nms3_Notre10_H5x5[:,:] < np.max(nms3_Notre10_H5x5) * 0.01] = 0
notre10_nms3_h5x5_final = overlay_img(Notre_original_image, nms3_Notre10_H5x5, 0.85 ,0.15)
#cv2.imwrite(os.path.join(currentDir,'','1','output', 'normal','d_Notre10_NMS3_H5x5.jpg'), notre10_nms3_h5x5_final)

nms5_Notre10_H5x5 = non_max_suppression(harris_response_Notre10_5x5, 5)
nms5_Notre10_H5x5[nms5_Notre10_H5x5[:,:] < np.max(nms5_Notre10_H5x5) * 0.01] = 0
notre10_nms5_h5x5_final = overlay_img(Notre_original_image, nms5_Notre10_H5x5, 0.85 ,0.15)
#cv2.imwrite(os.path.join(currentDir,'','1','output', 'normal','d_Notre10_NMS5_H5x5.jpg'), notre10_nms5_h5x5_final)


## ----------------------------- finish normal--------------------- ##
### Transformed

def rotate_img(img, rotate_angle, scale):
    (h, w, d) = img.shape # 讀取圖片大小
    center = (w // 2, h // 2) # 找到圖片中心
    
    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, rotate_angle, scale)
    
    # 第三個參數變化後的圖片大小
    rotate_img = cv2.warpAffine(img, M, (w, h))
    
    return rotate_img

### B.a.Gaussian Blur

t_chess_original_image = rotate_img(chess_original_image, 30, 0.5)

t_Notre_original_image = rotate_img(Notre_original_image, 30, 0.5)

t_chess_gaussian_blur_5 = gaussian_BGR_blur(gaussian_norm_kernel_5, t_chess_original_image)
t_chess_gaussian_blur_10 = gaussian_BGR_blur(gaussian_norm_kernel_10, t_chess_original_image)
t_Notre_gaussian_blur_5 = gaussian_BGR_blur(gaussian_norm_kernel_5, t_Notre_original_image)
t_Notre_gaussian_blur_10 = gaussian_BGR_blur(gaussian_norm_kernel_10, t_Notre_original_image)

### B.b.Intensity Gradient
# Calculate magnitude and gradient direction
t_chess_sobel_MagAngle_Gauss5 = sobel_magnitude_angle(t_chess_gaussian_blur_5)
t_chess_sobel_MagAngle_Gauss10 = sobel_magnitude_angle(t_chess_gaussian_blur_10)
t_Notre_sobel_MagAngle_Gauss5 = sobel_magnitude_angle(t_Notre_gaussian_blur_5)
t_Notre_sobel_MagAngle_Gauss10 = sobel_magnitude_angle(t_Notre_gaussian_blur_10)

# Make magnitude and direction into HSV color style
t_chess_angle_hsv_Gauss5 = MagAngle_to_HSV(t_chess_sobel_MagAngle_Gauss5)
t_chess_angle_hsv_Gauss10 = MagAngle_to_HSV(t_chess_sobel_MagAngle_Gauss10)
t_Notre_angle_hsv_Gauss5 = MagAngle_to_HSV(t_Notre_sobel_MagAngle_Gauss5)
t_Notre_angle_hsv_Gauss10 = MagAngle_to_HSV(t_Notre_sobel_MagAngle_Gauss10)

# Make Magnitude images 
t_chess_sobel_magnitude_Gauss5 = t_chess_sobel_MagAngle_Gauss5[0]
t_chess_sobel_magnitude_Gauss10 = t_chess_sobel_MagAngle_Gauss10[0]
t_Notre_sobel_magnitude_Gauss5 = t_Notre_sobel_MagAngle_Gauss5[0]
t_Notre_sobel_magnitude_Gauss10 = t_Notre_sobel_MagAngle_Gauss10[0]

### B.c.Structure Tensor (with Gaussian kernel size=10)
t_chess_sobel_direction_xy_gaussian_10 = sobel_derivative_xy(t_chess_sobel_magnitude_Gauss10)  
# Threshold 28 seems good
t_threshold_of_Magnitude = 45
# Threshold function: pixel < threshold => 0, others remain the original value
ret, t_Notre_Magnitude_Thresh_size10 = cv2.threshold(t_Notre_sobel_magnitude_Gauss10, t_threshold_of_Magnitude, 255,cv2.THRESH_TOZERO) 
t_Notre_sobel_direction_xy_gaussian_10 = sobel_derivative_xy(t_Notre_Magnitude_Thresh_size10) 
# Turn derivative into structure tensor of every pixel
t_structure_tensor_chess_gaussian_10 = structure_tensor_make(t_chess_sobel_direction_xy_gaussian_10)
t_structure_tensor_Notre_gaussian_10 = structure_tensor_make(t_Notre_sobel_direction_xy_gaussian_10)
# Calcualte SSD of tensors by window size
t_SSD_chess_gaussian_10_3x3 = SSD_make(t_structure_tensor_chess_gaussian_10,  3)
t_SSD_chess_gaussian_10_5x5 = SSD_make(t_structure_tensor_chess_gaussian_10,  5)
t_SSD_Notre_gaussian_10_3x3 = SSD_make(t_structure_tensor_Notre_gaussian_10,  3)
t_SSD_Notre_gaussian_10_5x5 = SSD_make(t_structure_tensor_Notre_gaussian_10,  5) # maybe we should use other threshold for 5x5 instead of 28
# Obtain small eigenvalue to know if it is an edge/corner
t_thresh_for_chess = 300
t_threshold_for_Notre = 300
t_small_eigenvalue_chess_window_3x3 = little_eigenvalue(t_SSD_chess_gaussian_10_3x3, t_thresh_for_chess)
t_small_eigenvalue_chess_window_5x5 = little_eigenvalue(t_SSD_chess_gaussian_10_5x5, t_thresh_for_chess)
t_small_eigenvalue_Notre_window_3x3 = little_eigenvalue(t_SSD_Notre_gaussian_10_3x3, t_threshold_for_Notre)
t_small_eigenvalue_Notre_window_5x5 = little_eigenvalue(t_SSD_Notre_gaussian_10_5x5, t_threshold_for_Notre)

### B.d Corner Detection with Non-Maximum Suppression 
t_harris_response_Notre10_3x3 = harris_response(t_SSD_Notre_gaussian_10_3x3, 100000)
t_harris_response_Notre10_5x5 = harris_response(t_SSD_Notre_gaussian_10_5x5, 100000)

t_nms3_chess10_H3x3 = non_max_suppression(t_small_eigenvalue_chess_window_3x3, 3)
t_nms3_chess10_H3x3[t_nms3_chess10_H3x3[:,:] < np.max(t_nms3_chess10_H3x3) * 0.005] = 0
t_chess10_nms3_h3x3_final = overlay_img(t_chess_original_image, t_nms3_chess10_H3x3, 0.95 ,0.1)
# cv2.imwrite(os.path.join(currentDir,'','1','output', 'transformed','d_chess10_NMS3_H3x3.jpg'), t_chess10_nms3_h3x3_final)

t_nms5_chess10_H3x3 = non_max_suppression(t_small_eigenvalue_chess_window_3x3, 5)
t_nms5_chess10_H3x3[t_nms5_chess10_H3x3[:,:] < np.max(t_nms5_chess10_H3x3) * 0.005] = 0
t_chess10_nms5_h3x3_final = overlay_img(t_chess_original_image, t_nms5_chess10_H3x3, 0.95 ,0.1)
# cv2.imwrite(os.path.join(currentDir,'','1','output', 'transformed','d_chess10_NMS5_H3x3.jpg'), t_chess10_nms5_h3x3_final)

t_nms3_chess10_H5x5 = non_max_suppression(t_small_eigenvalue_chess_window_5x5, 3)
t_nms3_chess10_H5x5[t_nms3_chess10_H5x5[:,:] < np.max(t_nms3_chess10_H5x5) * 0.01] = 0
t_chess10_nms3_h5x5_final = overlay_img(t_chess_original_image, t_nms3_chess10_H5x5, 0.95 ,0.15)
# cv2.imwrite(os.path.join(currentDir,'','1','output', 'transformed','d_chess10_NMS3_H5x5.jpg'), t_chess10_nms3_h5x5_final)

t_nms5_chess10_H5x5 = non_max_suppression(t_small_eigenvalue_chess_window_5x5, 5)
t_nms5_chess10_H5x5[t_nms5_chess10_H5x5[:,:] < np.max(t_nms5_chess10_H5x5) * 0.01] = 0
t_chess10_nms5_h5x5_final = overlay_img(t_chess_original_image, t_nms5_chess10_H5x5, 0.95 ,0.15)
# cv2.imwrite(os.path.join(currentDir,'','1','output', 'transformed','d_chess10_NMS5_H5x5.jpg'), t_chess10_nms5_h5x5_final)

t_nms3_Notre10_H3x3 = non_max_suppression(t_harris_response_Notre10_3x3, 3)
t_nms3_Notre10_H3x3[t_nms3_Notre10_H3x3[:,:] < np.max(t_nms3_Notre10_H3x3) * 0.01] = 0
t_notre10_nms3_h3x3_final = overlay_img(t_Notre_original_image, t_nms3_Notre10_H3x3, 0.85 ,0.15)
# cv2.imwrite(os.path.join(currentDir,'','1','output', 'transformed','d_Notre10_NMS3_H3x3.jpg'), t_notre10_nms3_h3x3_final)

t_nms5_Notre10_H3x3 = non_max_suppression(t_harris_response_Notre10_3x3, 5)
t_nms5_Notre10_H3x3[t_nms5_Notre10_H3x3[:,:] < np.max(t_nms5_Notre10_H3x3) * 0.01] = 0
t_notre10_nms5_h3x3_final = overlay_img(t_Notre_original_image, t_nms5_Notre10_H3x3, 0.85 ,0.15)
# cv2.imwrite(os.path.join(currentDir,'','1','output', 'transformed','d_Notre10_NMS5_H3x3.jpg'), t_notre10_nms5_h3x3_final)

t_nms3_Notre10_H5x5 = non_max_suppression(t_harris_response_Notre10_5x5, 3)
t_nms3_Notre10_H5x5[t_nms3_Notre10_H5x5[:,:] < np.max(t_nms3_Notre10_H5x5) * 0.01] = 0
t_notre10_nms3_h5x5_final = overlay_img(t_Notre_original_image, t_nms3_Notre10_H5x5, 0.85 ,0.15)
# cv2.imwrite(os.path.join(currentDir,'','1','output', 'transformed','d_Notre10_NMS3_H5x5.jpg'), t_notre10_nms3_h5x5_final)

t_nms5_Notre10_H5x5 = non_max_suppression(t_harris_response_Notre10_5x5, 5)
t_nms5_Notre10_H5x5[t_nms5_Notre10_H5x5[:,:] < np.max(t_nms5_Notre10_H5x5) * 0.01] = 0
t_notre10_nms5_h5x5_final = overlay_img(t_Notre_original_image, t_nms5_Notre10_H5x5, 0.85 ,0.15)
# cv2.imwrite(os.path.join(currentDir,'','1','output', 'transformed','d_Notre10_NMS5_H5x5.jpg'), t_notre10_nms5_h5x5_final)

## ---------------------------- finish transformed ----------------- ##
