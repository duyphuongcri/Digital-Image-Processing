import cv2 
import numpy as np 
import matplotlib.pyplot as plt

def histogram_equalization(imgage_input):
    hist,bins = np.histogram(imgage_input.flatten(),256,[0,256])
    cdf = hist.cumsum() 
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img_equalization = cdf[imgage_input]  
    return img_equalization, cdf
def histogram_matching(original, specified):
    img_equalization, cdf_original = histogram_equalization(original)
    _, cdf_specified = histogram_equalization(specified)
    array_convert = np.zeros((256), dtype=np.uint8)
    old_value = 0
    cdf_original_list,  cdf_specified_list = cdf_original.tolist(), cdf_specified.tolist()
    for i in range(256):
        if cdf_original_list[i] in cdf_specified_list:
            array_convert[i] = cdf_specified_list.index(cdf_original_list[i])
            old_value = cdf_specified_list.index(cdf_original_list[i])
        else:
            array_convert[i] = old_value
    img_matching = array_convert[original]
    return img_equalization, img_matching
#############################################
if __name__=='__main__':
    img = cv2.imread("lena.jpeg")
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_reference = img[:,:,0] + 50
    img_equalization, img_matching = histogram_matching(img_input, img_reference) 
    ouput = np.hstack((img_input, img_equalization, img_reference, img_matching)) 
    cv2.imshow('', ouput)
    cv2.waitKey(0)
cv2.destroyAllWindows()
