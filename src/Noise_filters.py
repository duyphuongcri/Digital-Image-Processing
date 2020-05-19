import numpy as np 
import cv2 
import matplotlib.pyplot as plt

def median_filter(image, size_kernel=3):
    # Apply for salt and pepper noise 
    h, w = image.shape
    output = np.zeros_like(image)
    for i in range(int(h - size_kernel + 1)):
        for j in range(int(w - size_kernel + 1)):
            sub_image = image[i: i + size_kernel, j: j + size_kernel]
            sub_image = sorted(list(sub_image.reshape(1, -1)[0]))
            #output[i: i + size_kernel, j: j + size_kernel] = sub_image[int((size_kernel ** 2 - 1) / 2)]
            output[i + int((size_kernel - 1)/2), j + int((size_kernel - 1)/2)] = sub_image[int((size_kernel ** 2 - 1) / 2)]
    return output

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
if __name__=='__main__':
    img = cv2.imread("moon.jpg", 0)
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    img_salt_pepper = add_salt_and_pepper_noise(img.copy(), 0.01)
    img_median_filter = median_filter(img_salt_pepper, size_kernel=3)

    cv2.imshow("Original ", img)
    cv2.imshow("salt and pepper", img_salt_pepper)
    cv2.imshow("median filter", img_median_filter)

    cv2.waitKey(0) 
cv2.destroyAllWindows()