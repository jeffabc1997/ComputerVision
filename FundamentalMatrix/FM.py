import numpy as np
import matplotlib.pyplot as plt
import cv2

def compute_fundamental_matrix(homo_coordinate_left, homo_coordinate_right):
    # homo_coor size: n x 3
    num_of_point = homo_coordinate_left.shape[0]
    # make correspondance matrix by (u, v, 1) x (u', v', 1)
    kronecker_coordinate = np.zeros((num_of_point,9)) # n x 9, result of kronecker product
    for i in range(num_of_point):
        kronecker_coordinate[i] = np.kron(homo_coordinate_left[i], homo_coordinate_right[i]) 
    # SVD of correspondance to find nullspace
    U_corres, D_corres, V_corres = np.linalg.svd(kronecker_coordinate, full_matrices=False) 
    Fundamental_raw = V_corres[-1].reshape((3, 3)) # choose the vector that corresponds to the smallest eigenvalue
    U_raw, D_raw, V_raw = np.linalg.svd(Fundamental_raw, full_matrices=False) # SVD to reconstruct Fundamental matrix
    D_raw[-1] = 0 # make the smallest eigenvalue to 0: rank-2 constraint

    Fundamental_matrix = np.matmul(np.matmul(U_raw, np.diag(D_raw)), V_raw) # recontruct with rank 2
    
    return Fundamental_matrix / Fundamental_matrix[2, 2]

def compute_normalized_fundamental_matrix(homo_coordinate_left, homo_coordinate_right):
    # homo_coordinate size: n x 3
    num_of_point = homo_coordinate_left.shape[0]
    # Hartley's approach
    ## -- scale the mean square distance --
    mean_m1 = np.mean(homo_coordinate_left, axis = 0) # axis = 0 so that m = (mean(x), mean(y), 1)
    mean_m2 = np.mean(homo_coordinate_right, axis = 0) # m_left
    
    distance_square1 = np.square(homo_coordinate_left - mean_m1)  # [(xi - mean(x)) ^2, (yi - mean(y)) ^2]
    distance_square2 = np.square(homo_coordinate_right - mean_m2)
    s1 = np.sqrt(np.sum(distance_square1) / (2*num_of_point)) # scale_left = summation of (all the square / number of points)
    s2 = np.sqrt(np.sum(distance_square2) / (2*num_of_point)) # scale_right
    T1 = np.array([[1/s1, 0, -(mean_m1[0] / s1)], [0, 1/s1, -(mean_m1[1]/s1)], [0, 0, 1]])
    T2 = np.array([[1/s2, 0, -(mean_m2[0] / s2)], [0, 1/s2, -(mean_m2[1]/s2)], [0, 0, 1]])
    
    normalized_coordinate1 = np.matmul(T1, np.transpose(homo_coordinate_left)) # 3 x n
    normalized_coordinate2 = np.matmul(T2, np.transpose(homo_coordinate_right))
    ## -- get scaled -- 
    # calculate fundamental matrix, input n x 3
    normalized_Fundamental_matrix = compute_fundamental_matrix(np.transpose(normalized_coordinate1), np.transpose(normalized_coordinate2))
    # reverse to original scale
    F = np.matmul(np.transpose(T1),np.matmul(normalized_Fundamental_matrix,T2)) 
    F = F / F[2,2] # set F[2,2] = 1
    return F

def calculate_right_epipolar_line(f_matrix, homo_coordinate_left): # will draw on the right image
    homo_vertical = np.transpose(homo_coordinate_left) # input: nx3 output: 3xn
    right_line_coefficient = np.matmul(np.transpose(f_matrix), homo_vertical)
    return right_line_coefficient # output 3 x n

def calculate_left_epipolar_line(f_matrix, homo_coordinate_right):
    homo_vertical = np.transpose(homo_coordinate_right) # input: nx3 output: 3xn
    left_line_coefficient = np.matmul(f_matrix, homo_vertical) 
    return left_line_coefficient # output 3 x n

def plot_epiploar_line(line_coefficient, homo_coordinate, image,file_name):
    num_points = homo_coordinate.shape[0]
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(image).astype('uint8')) 
    
    x_line = np.array([0.0, 512.0] * num_points) # start index and end index of x
    y_line = np.zeros((num_points,2)) # shape: n x 2
    x = np.array([0, 512])
    for i in range(num_points):
        a, b, c = line_coefficient[:,i].ravel()
        # to calculate where to draw on the image
        y_line[i] = -(x*a + c) / b # ax + by + c = 0
    x_line_T = np.transpose(x_line) # 2 x n [0]: start, [1]: endpoint
    y_line_T = np.transpose(y_line)
    # plot([[x1_start, x2_start,...],[x1_end, x2_end,...]], [[y1_start, y2_start, ...],[y1_end, y2_end, ...]])
    ax.plot([x_line_T[0], x_line_T[1]], [y_line_T[0], y_line_T[1]],linewidth=1) # plot epipolar line
    homo_vertical = np.transpose(homo_coordinate) # input: nx3, output: 3xn
    ax.plot(homo_vertical[0], homo_vertical[1], 'xr', markersize = 5) # mark corresponding points
    plt.ylim([512, 0]) # limit the canvas size
    plt.savefig(f"./output/{file_name}.png", dpi = 300)
    #plt.show()

def average_loss_distance(line_coefficient, homo_coordinate):
    # homo_coor shape: n x 3
    num_points = homo_coordinate.shape[0]
    distance_1point = np.zeros((num_points)).astype(np.float64)
    # ||ax+by+c|| / (a**2+b**2)**0.5
    for i in range(num_points):
        a, b, c = line_coefficient[:,i].ravel() # input 3 x n
        distance_1point[i] = abs(a * homo_coordinate[i][0] + b * homo_coordinate[i][1] + c) / np.sqrt((a**2 + b**2))
    distance_all = np.sum(distance_1point)
    return distance_all / num_points


with open('pt_2D_1.txt', 'r') as f:
    num_points = int(f.readline()) # n points
    coordinate_left = [[float(num) for num in x_y.split(' ')] for x_y in f]

with open('pt_2D_2.txt', 'r') as f:
    next(f)
    coordinate_right = [[float(num) for num in x_y.split(' ')] for x_y in f]

homo_coordinate_left = cv2.convertPointsToHomogeneous(np.asarray(coordinate_left)).reshape((num_points, 3)) # n x 3
homo_coordinate_right = cv2.convertPointsToHomogeneous(np.asarray(coordinate_right)).reshape((num_points, 3))

# part a
fundamental_matrix = compute_fundamental_matrix(homo_coordinate_left, homo_coordinate_right)
print("(a) Fundamental Matrix:\n", fundamental_matrix)

house1 = cv2.imread("image1.jpeg")
house2 = cv2.imread("image2.jpeg")

left_epipolar_line = calculate_left_epipolar_line(fundamental_matrix, homo_coordinate_right)
right_epipolar_line = calculate_right_epipolar_line(fundamental_matrix,homo_coordinate_left)
plot_epiploar_line(left_epipolar_line, homo_coordinate_left, house1, "a_img1") # plot and save
plot_epiploar_line(right_epipolar_line, homo_coordinate_right, house2, "a_img2")

left_distance = average_loss_distance(left_epipolar_line, homo_coordinate_left)
right_distance = average_loss_distance(right_epipolar_line, homo_coordinate_right)
print("(a) Average Distance: ", left_distance + right_distance)

# part b
normalized_fundamental_matrix = compute_normalized_fundamental_matrix(homo_coordinate_left, homo_coordinate_right)
print("(b) Normalized Fundamental Matrix:\n", normalized_fundamental_matrix)

left_epipolar_line = calculate_left_epipolar_line(normalized_fundamental_matrix, homo_coordinate_right)
right_epipolar_line = calculate_right_epipolar_line(normalized_fundamental_matrix,homo_coordinate_left)
plot_epiploar_line(left_epipolar_line, homo_coordinate_left, house1, "b_img1") # plot and save
plot_epiploar_line(right_epipolar_line, homo_coordinate_right, house2, "b_img2")

left_distance = average_loss_distance(left_epipolar_line, homo_coordinate_left)
right_distance = average_loss_distance(right_epipolar_line, homo_coordinate_right)
print("(b) Average Distance: ", left_distance + right_distance)