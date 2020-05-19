"""
1. Generate an image with motion.
2. Display the original image and the motion blurred image
3. Recover the original image from the motion blurred image. You can use whatever method but it should be the simplest. 
    Maybe inverser filtering can work.
4. Then add Gaussian noise to the motion blurred image. Display the motion blurred image with Gaussian image.
5. Try to recover the original image from the motion blurring and the Gaussian noise. 
    Compare inverser filtering and Wiener filtering. 
    Maybe you have to try other methods or devise a new method of your own.
"""
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from skimage import restoration
import scipy.fftpack as fp
def apply_motion_blurring(img):
    FM = fp.fft2(img)
    H = kernel(img)
    H = fp.fft2(fp.ifftshift(H))
    GM = FM*H
    G = fp.ifft2(GM).real
    return G.astype(np.uint8)

def inverse_filter(blur_img):
    G = fp.fft2(blur_img )
    H = kernel(blur_img)
    H = fp.fft2(fp.ifftshift(H))
    epsilon = 0.01
    H = 1 / (epsilon + H)
    F_hat = G*H
    f_hat = fp.ifft2(F_hat).real
    return f_hat

def add_gaussian_noise(image):
    mean, sigma = 0, 20 # mean and standard deviation
    noisy = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    img_noise = image.astype(np.float32) + noisy
    img_noise[img_noise > 255] = 255
    img_noise[img_noise < 0] = 0
    return img_noise.astype(np.uint8)

def wiener_filter(img):
    G =  fp.fft2(img)
    H = kernel(img)
    f_hat = restoration.wiener(img, H, 5, clip=False)
    return f_hat

def kernel(img):
    size = 15
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    kernel = np.pad(kernel, (((img.shape[0]-size)//2,(img.shape[0]-size)//2+1), ((img.shape[1]-size)//2,(img.shape[1]-size)//2+1)), "constant", constant_values=(0,0))
    return kernel
if __name__=='__main__':
    img = cv2.imread("./images/aerial_view_no_turb.tif", 0)
    motion_blurring_image = apply_motion_blurring(img)
    motion_blur_gassian_image = add_gaussian_noise(motion_blurring_image)
    inverse_filter_img = inverse_filter(motion_blurring_image)
    # # Apply Wiener Filter
    inverse_filter2 = inverse_filter(motion_blur_gassian_image)
    wiener_filter_img = wiener_filter(motion_blur_gassian_image)
    # # Display results
    display = [img, motion_blurring_image, motion_blur_gassian_image,inverse_filter_img, wiener_filter_img, inverse_filter2]
    label = ['Original Image', 'Motion Blurred Image', 'Motion Blurring + Gaussian Noise', 'Inverse Filter Applied' ,'Wiener Filter applied', 'Inverse Filter Applied 2']

    fig = plt.figure(figsize=(20, 10))

    for i in range(len(display)):
        fig.add_subplot(2, 3, i+1)
        plt.imshow(display[i], cmap = 'gray')
        plt.title(label[i])

    plt.show()

cv2.destroyAllWindows()