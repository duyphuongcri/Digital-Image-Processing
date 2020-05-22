"""
Apply kmeans clustering into color image
"""
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import cdist

def kmeans_clustering(image, K):
    X = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
    centers = X[np.random.choice(X.shape[0], K, replace=False)]
    labels = []
    i = 0
    while True:
        # calculate  distances about value between pixels and centers
        D = cdist(X, centers)    
        labels = np.argmin(D, axis = 1)
        new_centers = kmeans_update_centers(X, labels, K)
        if np.any(new_centers == centers) == True:
            break
        centers = new_centers
        i += 1

    output = np.zeros_like(X) 
    for k in range(K):
        output[labels == k] = centers[k]
    output = output.reshape(image.shape)
    return output

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster 
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers.astype(np.uint8)

if __name__=="__main__":
    K = 2
    image = cv2.imread("./images/cr7.jpg")
    # RGB color space
    output_RGB = kmeans_clustering(image, K)

    # HSV color space
    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    output_HSV = kmeans_clustering(HSV_image, K)

    # HLS color space
    HLS_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    output_HLS = kmeans_clustering(HLS_image, K)

    # Lab color space
    Lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    output_Lab = kmeans_clustering(Lab_image, K)

    # Lab color space
    YCrCb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    output_YCrCb = kmeans_clustering(YCrCb_image, K)
    # # Display results
    display = [image, output_RGB, output_HSV, output_HLS, output_Lab, output_YCrCb]
    label = ['Original Image', 'Kmeans Clustering with RGB', 'Kmeans Clustering with HSV', 'Kmeans Clustering with HLS' ,'Kmeans Clustering with Lab','Kmeans Clustering with YCrCb']
    fig = plt.figure(figsize=(18, 10))

    for i in range(len(display)):
        fig.add_subplot(2, 3, i+1)
        plt.imshow(display[i], cmap = 'gray')
        plt.title(label[i])

    plt.show()