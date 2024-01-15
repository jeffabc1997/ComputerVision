import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import time
def getpixel(image,x, y):
    rgb = image[x][y][0], image[x][y][1], image[x][y][2]
    return rgb
def euclidean_dis_rgb(arr1, arr2):
    arr1 = arr1[:3]
    arr2 = arr2[:3]
    diff = arr1 - arr2
    diff_square = np.square(diff)
    dist_Sumofsquare = np.sum(diff_square)
    return np.sqrt(dist_Sumofsquare)

def plot_rgb_original(data, bandwidth, output_folder, output_name):
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    np.unique(data, axis=0)
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c = data/255)
    ax.set_xlabel('X label: Red')
    ax.set_ylabel('Y label: Green')
    ax.set_zlabel('Z label: Blue')
    # plt.savefig(f"./output/{output_folder}/2C_rgb_Original_{output_name}.jpg", dpi = 300)

def plot_rgb_distribution(data, color_data, iteration, bandwidth, output_folder, output_name):
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    np.unique(data, axis=0)
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c = color_data/255)
    ax.set_xlabel('X label: Red')
    ax.set_ylabel('Y label: Green')
    ax.set_zlabel('Z label: Blue')
    # plt.savefig(f"./output/{output_folder}/2C_rgb_{output_name}_{iteration}_band{bandwidth}.jpg", dpi = 300)

def k_means(path, k_cluster, iterations, output_folder, filename):
    
    img = cv2.imread(path)
    height, width = img.shape[:2]
    #print(height, width)
    pixelClusterAppartenance = np.ndarray(img.shape[:2], dtype=int)
    dataVector = np.ndarray(shape=(height, width, 5), dtype=np.float32)
    
    for x in range(0, height):
        for y in range(0, width):     
            rgb = getpixel(img, x, y)
            dataVector[x, y, 0] = rgb[0]
            dataVector[x, y, 1] = rgb[1]
            dataVector[x, y, 2] = rgb[2]
            dataVector[x, y, 3] = x
            dataVector[x, y, 4] = y
    np.random.seed(int(time.time())) # real random
    x_ci = np.random.randint(0, height, k_cluster) # random center
    y_ci = np.random.randint(0, width, k_cluster)

    centers = np.ndarray((k_cluster, 5), dtype= np.float32)
    old_center = np.ndarray((k_cluster, 5), dtype= np.float32)
    for k in range(k_cluster):
        rgb = img[x_ci[k]][y_ci[k]][:3]
        centers[k] = (rgb[0], rgb[1], rgb[2], x_ci[k], y_ci[k]) # set centroids' rgb and coordinate

    loss_array = np.zeros(img.shape[:2])
    
    for i in range(iterations):
        dataInCenter = []
        for k in range(k_cluster):
            dataInCenter.append([])
        # calculate distance of each pixel from each center
        for x in range(height):
            for y in range(width):
                distanceToCenters = np.ndarray(shape=(k_cluster))
                for k in range(k_cluster): 
                    distanceToCenters[k] = euclidean_dis_rgb(dataVector[x][y], centers[k])
                
                pixelClusterAppartenance[x][y] = np.argmin(distanceToCenters) # assign pixel to that cluster
                loss_array[x][y] = distanceToCenters[pixelClusterAppartenance[x][y]]

        # make each pixel in a cluster
        for x in range(height):
            for y in range(width):              
                for k in range(k_cluster):
                    if pixelClusterAppartenance[x][y] == k: # if that pixel belongs to cluster K
                        dataInCenter[k].append(dataVector[x][y])
        #print("find new center: ")
        for k in range(k_cluster):
            dataOneCluster = np.array(dataInCenter[k])
            old_center[k] = centers[k]
            centers[k] = np.mean(dataOneCluster, axis = 0) # mean of each cluster's center's rgb and xy
            #print(k, ": ",centers[k][3], " ",centers[k][4])
        diff = np.absolute(centers[:,3:] - old_center[:, 3:])
        diff = np.sum(diff)
        conv_threshold = k_cluster * 0.4 # set threshold for the centers alternation
        #print("diff: ", diff)
        if diff < conv_threshold:
            break
    # compute objective function
    loss = np.sum(loss_array)
    # for x in range(height):
    #     for y in range(width):
    #         loss += euclidean_dis_rgb(dataVector[x][y], centers[pixelClusterAppartenance[x][y]])
    output_image = np.zeros(img.shape)
    for x in range(height):
        for y in range(width):
            output_image[x][y] = centers[pixelClusterAppartenance[x][y]][:3]
    #print(centers)
    # cv2.imwrite(f"./output/{output_folder}/2A_Kmeans_{filename}_k{k_cluster}.jpg", output_image)
    return loss

def k_means_pp_center(path, k_cluster):

    img = cv2.imread(path)
    height, width = img.shape[:2]

    dataVector = np.ndarray(shape=(height* width, 5), dtype=np.float32)     
    for x in range(0, height):
        for y in range(0, width):     
            rgb = getpixel(img, x, y)
            dataVector[y + x * width, 0] = rgb[0]
            dataVector[y + x * width, 1] = rgb[1]
            dataVector[y + x * width, 2] = rgb[2]
            dataVector[y + x * width, 3] = x
            dataVector[y + x * width, 4] = y
            
    np.random.seed(int(time.time())) # real random

    x_ci = np.random.randint(0, height*width) # random initial center
    centers = np.ndarray((k_cluster, 5), dtype= np.float32)
    centers[0] = (dataVector[x_ci,0], dataVector[x_ci,1], dataVector[x_ci,2], dataVector[x_ci,3], dataVector[x_ci,4])
    #print("center 0:", centers[0])
    distance = 0
    distance_prop = np.zeros(shape=(height* width), dtype=np.float32) # probability based on distance
    for i in range(k_cluster-1):
        for x in range(0, height* width):
            
            distance = 0 # distance of 1 pixel            
            for j in range(i+1):
                distance += euclidean_dis_rgb(dataVector[x], centers[j]) # total distance from the existing centers
            distance_prop[x] = distance
        
        [potential_center] = random.choices(dataVector, weights=distance_prop[:,]) # origianl random.choices() return a list

        #print("potential center: ",potential_center)
        center_x = int(potential_center[3]) # coordinate in the input image
        center_y = int(potential_center[4])

        centers[i+1] = (img[center_x,center_y,0], img[center_x,center_y,1],img[center_x,center_y,2], center_x, center_y)  
         
    return centers
  
def k_means_Plus(path, k_cluster, iterations, output_folder, filename):
    
    img = cv2.imread(path)
    height, width = img.shape[:2]

    #print(height, width)
    pixelClusterAppartenance = np.ndarray(img.shape[:2], dtype=int)
    dataVector = np.ndarray(shape=(height, width, 5), dtype=np.float32)
    
    for x in range(0, height):
        for y in range(0, width):     
            rgb = getpixel(img, x, y)
            dataVector[x, y, 0] = rgb[0]
            dataVector[x, y, 1] = rgb[1]
            dataVector[x, y, 2] = rgb[2]
            dataVector[x, y, 3] = x
            dataVector[x, y, 4] = y

    centers = k_means_pp_center(path, k_cluster)
    old_center = np.ndarray((k_cluster, 5), dtype= np.float32)

    loss_array = np.zeros(img.shape[:2])

    for i in range(iterations):
        dataInCenter = []
        for k in range(k_cluster):
            dataInCenter.append([])
        # calculate distance of each pixel from each center
        
        for x in range(height):
            for y in range(width):
                distanceToCenters = np.ndarray(shape=(k_cluster))
                for k in range(k_cluster): 
                    distanceToCenters[k] = euclidean_dis_rgb(dataVector[x][y], centers[k])
                
                pixelClusterAppartenance[x][y] = np.argmin(distanceToCenters) # assign pixel to that cluster
                loss_array[x][y] = distanceToCenters[pixelClusterAppartenance[x][y]]

        # make each pixel in a cluster
        for x in range(height):
            for y in range(width):              
                for k in range(k_cluster):
                    if pixelClusterAppartenance[x][y] == k: # if that pixel belongs to cluster K
                        dataInCenter[k].append(dataVector[x][y])
        #print("find new center: ")
        for k in range(k_cluster):
            dataOneCluster = np.array(dataInCenter[k])
            old_center[k] = centers[k]
            centers[k] = np.mean(dataOneCluster, axis = 0) # mean of each cluster's center's rgb and xy

        diff = np.absolute(centers[:,3:] - old_center[:, 3:])
        diff = np.sum(diff)
        conv_threshold = k_cluster * 0.4 # set threshold for the centers alternation

        if diff < conv_threshold:
            break
    # compute objective function
    loss = np.sum(loss_array)

    output_image = np.zeros(img.shape)
    for x in range(height):
        for y in range(width):
            output_image[x][y] = centers[pixelClusterAppartenance[x][y]][:3]

    # cv2.imwrite(f"./output/{output_folder}/2B_kmeansPP_{filename}_k{k_cluster}.jpg", output_image)
    return loss

def uniform_mean_shift_3d(path, bandwidth, converge_count, output_folder, output_name):
    data = cv2.imread(path)
    
    height, width = data.shape[:2]
    dataVector = np.ndarray(shape=(height* width//2 , 3), dtype=np.float32)
    dataVector_len = height*width//2
    output_datavector = np.zeros(shape = (height*width ,3), dtype=np.float32)
    for x in range(0, height):
        for y in range(0, width//2):     
            rgb = getpixel(data, x, 2*y)
            dataVector[y + x * width//2, 0] = rgb[0]
            dataVector[y + x * width//2, 1] = rgb[1]
            dataVector[y + x * width//2, 2] = rgb[2]
    bandwidth_str = str(bandwidth)
    iteration_str = str(converge_count)
    plot_rgb_original(dataVector, bandwidth_str, output_folder, output_name)

    for i in range(0, dataVector_len):
        cluster_centroid = dataVector[i] # point that we focused on
        # Search points in circle
        for count in range(converge_count):
                  
            diff = np.sum(np.square(dataVector - cluster_centroid), axis = 1)
            indices = diff < bandwidth
            new_centroid = np.round(np.mean(dataVector[indices], axis = 0))
            if np.array_equal(new_centroid, cluster_centroid):             
                break
            # Update centroid     
            cluster_centroid = new_centroid # shift to the mean

        output_datavector[2*i:2*(i+1)] = cluster_centroid

        if (i%20000) == 0:
            print("order ",i)
    #print("output datavector: ",output_datavector.shape)
    output = output_datavector.reshape(data.shape).astype(np.float32)

    plot_rgb_distribution(dataVector, output_datavector, iteration_str, bandwidth, output_folder, output_name)
    
    # cv2.imwrite(f"./output/{output_folder}/2C_3d_{output_name}_it{iteration_str}_band{bandwidth_str}.jpg", output)

def uniform_mean_shift_5d(path, bandwidth, converge_count, output_folder, output_name):
    data = cv2.imread(path) / 255 # normalize to (0, 1)
    
    height, width = data.shape[:2]
    dataVector = np.ndarray(shape=(height* width//2, 5), dtype=np.float32)
    dataVector_len = height*width//2
    output_datavector = np.zeros(shape = (height * width, 5), dtype=np.float32)
    for x in range(0, height):
        for y in range(0, width//2):  
            rgb = getpixel(data, x, 2*y) ## aware of the 2*y
            dataVector[y + x * width//2, 0] = rgb[0]
            dataVector[y + x * width//2, 1] = rgb[1]
            dataVector[y + x * width//2, 2] = rgb[2]
            dataVector[y + x * width//2, 3] = x / height # normalize
            dataVector[y + x * width//2, 4] = y / width

    for i in range(0, dataVector_len):
        cluster_centroid = dataVector[i] # point that we focused on
        
        for count in range(converge_count): # Search points in circle           
            diff = np.sum(np.square(dataVector - cluster_centroid), axis = 1) # dimension: data points x 1
            indices = np.where(diff < bandwidth)[0]
            new_centroid = np.mean(dataVector[indices], axis = 0)
            #print("Diff: ", diff.shape)
            # Update centroid     
            old_centroid = cluster_centroid
            cluster_centroid = new_centroid # shift to the mean

            # old and new centroid the same, then break
            if np.array_equal(new_centroid, old_centroid):             
                break

        output_datavector[2*i:2*(i+1)] = cluster_centroid
        if (i%20000) == 0:
            print("order ",i)

    output_datavector255 = output_datavector[:, :3] * 255
    output = output_datavector255.reshape(data.shape).astype(np.float32)
    bandwidth_str = str(bandwidth)
    iteration_str = str(converge_count)
    plot_rgb_distribution(output_datavector255, iteration_str, bandwidth, output_folder, output_name)

    # cv2.imwrite(f"./output/{output_folder}/2D_5d_{output_name}_it{iteration_str}_band{bandwidth_str}.jpg", output)

def main():

    sunset = "2-image.jpg"
    image_folder = "image"
    sun_output = "image"

    buffalo = "2-masterpiece.jpg"
    masterpiece_folder = "masterpiece"
    buffalo_output = "masterpiece"
    
    initial_guess = 1
    k_cluster_num = [7]
    iterations = 13
    for k in k_cluster_num:
        for i in range(initial_guess):
            iteration_guess = str(i) # iteration 0, 1, 2...
            sun_output_several = sun_output + str(iteration_guess)
            master_output_several = buffalo_output + str(iteration_guess)

            k_means_Plus(sunset, k, iterations, image_folder, sun_output_several)
            k_means_Plus(buffalo, k, iterations, masterpiece_folder, master_output_several)
            k_means(sunset, k, iterations, image_folder, sun_output_several)
            k_means(buffalo, k, iterations, masterpiece_folder, master_output_several)
            
    uniform_mean_shift_3d(sunset, 550, 5, image_folder, sun_output)
    uniform_mean_shift_5d(sunset, 0.019, 5, image_folder, sun_output)
    uniform_mean_shift_3d(buffalo, 475, 5, masterpiece_folder, buffalo_output)
    uniform_mean_shift_5d(buffalo, 0.015, 5, masterpiece_folder, buffalo_output)

if __name__ == '__main__':
    main()


