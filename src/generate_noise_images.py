import numpy as np 
import cv2 
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, prob):
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.rand(1)
            if  rdn < prob:
                image[i][j] = 0
            elif rdn > thres:
                image[i][j] = 255
    return image
    
def add_gaussian_noise(image):
    mean, sigma = 0, 64 # mean and standard deviation
    noisy = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    img_noise = image.astype(np.float32) + noisy
    img_noise[img_noise > 255] = 255
    img_noise[img_noise < 0] = 0
    #plt.hist(noisy.ravel(),256,[-128,128]); plt.show()
    return img_noise.astype(np.uint8)

def add_uniform_noise(image, low, high):
    noisy = np.random.uniform(low, high, image.shape)
    img_noise = image.astype(np.float32) + noisy.astype(np.float32)
    img_noise[img_noise > 255] = 255
    #plt.hist(noisy.astype(np.uint8).ravel(),256,[0,256]); plt.show()
    return img_noise.astype(np.uint8)

if __name__=='__main__':
    img = cv2.imread("moon.jpg", 0)
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    img_salt_pepper = add_salt_and_pepper_noise(img.copy(), 0.01)
    img_gaussian_noise = add_gaussian_noise(img.copy())
    img_uniform_noise = add_uniform_noise(img.copy(), 50, 150)

    cv2.imshow("Original ", img)
    cv2.imshow("salt and pepper", img_salt_pepper)
    cv2.imshow("gaussian", img_gaussian_noise)
    cv2.imshow("uniform", img_uniform_noise)
    cv2.waitKey(0) 
cv2.destroyAllWindows()