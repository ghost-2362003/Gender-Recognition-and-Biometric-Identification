import torch
import numpy as np
import cv2
from guided_filter_pytorch.guided_filter import GuidedFilter

def createLowFrequencyComponent(img, guided_filter_Radius = 10):
    
    image = cv2.imread(img)
    #print(type(image))
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    gray_tensor = torch.from_numpy(grayscale_image).float().unsqueeze(0).unsqueeze(0) / 255.0

    # Use the already defined hr_x (GuidedFilter instance)
    GF = GuidedFilter(r=guided_filter_Radius, eps=0.01)
    
    low_freq_image = GF(gray_tensor, img_tensor)
    low_freq_image = low_freq_image.squeeze(0).permute(1, 2, 0)    ## convert tensor to proper image dimensions
    low_freq_image = low_freq_image.numpy()     ## convert tensor to numpy array
    
    #print("low freq image shape: ", low_freq_image.shape)
    #print("low freq image type: ", type(low_freq_image))
    
    return low_freq_image
    
def createHighFrequencyComponent(img, epsilon=0.01):
    
    image = cv2.imread(img)
    eps = np.full((1200, 1600, 3), epsilon)     ## for numerical stability
    eps_tensor = torch.from_numpy(eps).float().permute(0, 1, 2)     ## convert eps to tensor
    
    # create the low frequency image 
    low_freq_image = createLowFrequencyComponent(img)
    low_freq_image = torch.from_numpy(low_freq_image)       

    # create the high frequency image   
    high_frequency_image = image/(low_freq_image + eps_tensor)
    Ih_yuv = cv2.cvtColor(high_frequency_image.detach().numpy(), cv2.COLOR_RGB2YUV)
    Y = Ih_yuv[:, :, 0]
    high_frequency_image = (Y - Y.min()) / (Y.max() - Y.min())
    
    #print("high freq image shape: ", high_frequency_image.shape)
    #print("high freq image type: ", type(high_frequency_image))
    
    return high_frequency_image
    
#if __name__ == "__main__":
    
#    image = cv2.imread("G:/11k_hands/dataset/train/male/Hand_0000056.jpg")
    
#    createLowFrequencyComponent("G:/11k_hands/dataset/train/male/Hand_0000056.jpg")
#    createHighFrequencyComponent("G:/11k_hands/dataset/train/male/Hand_0000056.jpg")