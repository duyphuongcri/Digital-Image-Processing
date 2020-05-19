"""
1. For a series of input samples, display the result of Fourier transformation as shown in the Fig. 4.3 of the PPT file
2. Show the result of applying ideal low/high pass filter and Gaussian low/high pass filter
3. Select some sample images which demonstrates the effect of high frequency emphasis filtering.
4. Select some samples for homomorphic filtering and show the results. 
    Then perform the same algorithm without logarithm transformation and discuss what happens in such cases.
    Submit the code/result and the discussion on what happens in the sample images.
"""

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.fftpack as fp
from Historam import *
def distance(point1,point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base   
def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base

def homomorphicFilter(D0, imgShape):
    c = 1
    base = np.zeros(imgShape[:2], dtype=np.float64)
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - np.exp(c*((-distance((y,x),center)**2)/(2*(D0**2))))
    return base     

def frequenceEmphasisFiltering(D0, imgShape):
    base = np.zeros(imgShape[:2], dtype=np.float64)
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - np.exp((-distance((y,x),center)**2)/(2*(D0**2)))
    return base     
   
if __name__=="__main__":
    # image = np.zeros((512, 512), dtype=np.float64)
    # image[image.shape[0]//2 - 10:image.shape[0]//2 + 10, image.shape[1]//2 - 20:image.shape[1]//2 + 20] = 255
    image = cv2.imread("./images/a_letter.png", 0)
    freq_img = fp.fftshift(fp.fft2(image))
    #------------Allpy low and high pass filer--------------------------------
    lowPassFilter_img = abs(fp.ifft2(fp.ifftshift(freq_img*idealFilterLP(40, image.shape))))
    highPassFilter_img = abs(fp.ifft2(fp.ifftshift(freq_img*idealFilterHP(40, image.shape))))
    #----------- Apply gaussian low and high pass filter-------------------------------------------
    gaussianLowPassFilter_img = abs(fp.ifft2(fp.ifftshift(freq_img*gaussianLP(40, image.shape))))
    gaussianHighPassFilter_img = abs(fp.ifft2(fp.ifftshift(freq_img*gaussianHP(40, image.shape))))
    # # Display results
    display = [image, lowPassFilter_img, gaussianLowPassFilter_img, np.log(1+np.abs(freq_img)), highPassFilter_img, gaussianHighPassFilter_img]
    label = ['Original Image', "Processed Image with Low Pass Filter", "Processed Image with Gaussian Low Pass Filter", "Centered Fourier Spectrum", "Processed Image with High Pass Filter", "Processed Image with Gaussian High Pass Filter"]

    fig = plt.figure(figsize=(18, 8))

    for i in range(len(display)):
        fig.add_subplot(2, 3, i+1)
        plt.imshow(display[i], cmap = 'gray')
        plt.title(label[i])
    plt.show()

    #------------------------ Apply High Frequency Emphasis Filtering------------------------
    a, b = 0.5, 2
    A = 2
    image = cv2.imread("./images/lung.jpg", 0)
    freq_img = fp.fftshift(fp.fft2(image))
    H = frequenceEmphasisFiltering(30, image.shape)
    #H = fp.ifftshift(H)
    H = a + b*H
    filtered_img = freq_img*H
    img_hef = abs(fp.ifft2(fp.fftshift(filtered_img)))  # HFE filtering done
    img_equalization, cdf_original = histogram_equalization(img_hef.astype(np.uint8))

    # # Display results  
    display = [image, img_hef, img_equalization]
    label = ['Original Image', "Processed Image with HFE Filter", "Image after equaliztion" ]
    fig = plt.figure(figsize=(18, 8))
    for i in range(len(display)):
        fig.add_subplot(1, 3, i+1)
        plt.imshow(display[i], cmap = 'gray')
        plt.title(label[i])
    plt.show()    

    #--------------- Apply Homomorphic Filter---------------------------------
    gammaLow = 0.75
    gammaHigh = 2
    image = cv2.imread("./images/homo_ex.jpg", 0)
    image_log = np.log1p(image.astype(np.float32))
    freq_log_img = fp.fftshift(fp.fft2(image_log))
    H = homomorphicFilter(30, image.shape)
    H = fp.ifftshift(H)
    H = (gammaHigh - gammaLow)*H + gammaLow
    homomorphicFilter_img = H*freq_log_img
    homomorphicFilter_img = abs(fp.ifft2(homomorphicFilter_img))
    output = np.exp(homomorphicFilter_img)
    # # Display results  
    display = [image, output]
    label = ['Original Image', "Processed Image Homomorphic Filter" ]
    fig = plt.figure(figsize=(18, 8))
    for i in range(len(display)):
        fig.add_subplot(1, 2, i+1)
        plt.imshow(display[i], cmap = 'gray')
        plt.title(label[i])
    plt.show()