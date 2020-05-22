"""
Author: Duy-Phuong Dao
Email: duyphuongcri@gmail.com
"""

import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
def calculate_gradient(image):
    kernel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_y = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    gradient_x = cv2.filter2D(image, -1, kernel_x) # gradient_x = [Bx, Gx, Rx]
    gradient_y = cv2.filter2D(image, -1, kernel_y) # gradient_y = [By, Gy, Ry]

    g_xx = gradient_x[:, :, 0]**2 + gradient_x[:, :, 1]**2 + gradient_x[:, :, 2]**2
    g_yy = gradient_y[:, :, 0]**2 + gradient_y[:, :, 1]**2 + gradient_y[:, :, 2]**2
    g_xy = gradient_x[:, :, 0]*gradient_y[:, :, 0] + gradient_x[:, :, 1]*gradient_y[:, :, 1] + gradient_x[:, :, 2]*gradient_y[:, :, 2]
    angle = np.arctan2(2*g_xy, g_xx - g_yy) / 2
    img_magnitude = (0.5*((g_xx + g_yy) + (g_xx - g_yy)*np.cos(2*angle) + 2*g_xy*np.sin(2*angle)))**0.5

    gradient_R = (gradient_x[:, :, 2]**2 + gradient_y[:, :, 2]**2)**0.5
    gradient_G = (gradient_x[:, :, 1]**2 + gradient_y[:, :, 1]**2)**0.5
    gradient_B = (gradient_x[:, :, 0]**2 + gradient_y[:, :, 0]**2)**0.5

    img_sub =  (gradient_B + gradient_R + gradient_G) - img_magnitude

    return img_magnitude,  gradient_B + gradient_R + gradient_G, img_sub

if __name__=="__main__":
    image = cv2. imread("./images/lena.jpg")
    img_magnitude, gradient_sum_RGB, img_sub = calculate_gradient(image.astype(np.float64))
    
    # # Display results
    display = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB), img_magnitude, gradient_sum_RGB, img_sub]
    label = ['Original Image', 'The magnitude of gradient', 'Image with sum of gradients of 3 channels', "The difference between left image and above image"]
    fig = plt.figure(figsize=(18, 10))

    for i in range(len(display)):
        fig.add_subplot(2, 2, i+1)
        plt.imshow(display[i], cmap = 'gray')
        plt.title(label[i])

    plt.show()