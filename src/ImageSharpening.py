import numpy as np 
from skimage.filters import laplace
import cv2
import matplotlib.pyplot as plt 

def laplace_sharpening(img):
    return np.clip(laplace(img) + img, 0, 1)

if __name__=="__main__":

    image = cv2.imread('./images/lena.jpeg', 0)
    #sharpened_img = laplace_sharpening(image)


    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = np.clip(cv2.filter2D(image, -1, kernel), 0, 255)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_img2 = np.clip(cv2.filter2D(image, -1, kernel), 0, 255)

    # # Display results
    display = [image, sharpened_img, sharpened_img2]
    label = ['Original Image', 'Sharpened Image with center of mask = 5', 'Sharpened Image with center of mask = 9']

    fig = plt.figure(figsize=(18, 8))

    for i in range(len(display)):
        fig.add_subplot(2, 3, i+1)
        plt.imshow(display[i], cmap = 'gray')
        plt.title(label[i])

    plt.show()

