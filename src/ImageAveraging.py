import numpy as np 
import cv2 

def add_gaussian_noise(image):
    h, w = image.shape
    # Gaussian distribution parameters
    mean = 0
    sigma = 64
    gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = np.zeros(image.shape, np.float32)
    noisy_image = cv2.addWeighted(image.astype(np.float32), 1.0, gaussian, 1.0,0)
    noisy_image[noisy_image > 255] = 255
    noisy_image[noisy_image < 0] = 0
    return noisy_image

if __name__=='__main__':
    img = cv2.imread("galaxy.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    list_number_images = [1, 8, 32, 64, 128]
    list_denoised_images = []
    for n_images in list_number_images:
        img_noise_average = np.zeros(img.shape[:2], dtype=np.float32)
        for i in range(n_images):
            img_noise = add_gaussian_noise(img_gray)
            img_noise_average = img_noise_average + img_noise
        img_noise_average = (img_noise_average / n_images).astype(np.uint8)
        list_denoised_images.append(img_noise_average)
        
    output= np.vstack((np.hstack([img_gray, list_denoised_images[0]]), 
                        np.hstack([list_denoised_images[1], list_denoised_images[2]]), 
                        np.hstack([list_denoised_images[3], list_denoised_images[4]])))

    cv2.imshow('', cv2.resize(output, (int(output.shape[1]/2), int(output.shape[0]/2))))
    cv2.waitKey(0) 
cv2.destroyAllWindows()


