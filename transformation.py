import math
import numpy as np
from skimage import transform
import cv2 as cv
from scipy.ndimage.filters import gaussian_filter


def get_sobel_img(img):
    scale = 4
    delta = 0
    ddepth = cv.CV_16S

    grad_x = cv.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    print(gard_x)
    
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return grad

def noisy(noise_type, image, sig):
    if (noise_type == 0):
        row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var**sig
        gauss = np.random.normal(mean, sigma, (row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif (noise_type == 1):
        blurred = gaussian_filter(np.float32(image), sigma=sig)
        return blurred
    
def get_otsu_ed_img(img_np):
    otus_img = cv.threshold(img_np, 0, 255, cv.THRESH_OTSU)[1]
            
    erosion = cv.erode(otus_img, np.ones((3, 3), np.uint8), iterations = 1)
    dilation = cv.dilate(erosion, np.ones((3, 3), np.uint8), iterations = 3)

    dilation = np.where(dilation==1, 1, 0)
    
    return dilation

def shear_transform(seed, img_np):
    tform = transform.AffineTransform(
        shear=seed,
        )
    
    if (tform.params[0][1] < 0):
        factor = 256*(np.abs(tform.params[0][1]))
    else:
        factor = 0.
                
    M = [[1-np.abs(tform.params[0][1]), 0., factor],
         [0., 1., 0.],
         [0., 0., 1.]]
    
    img_np = cv.warpPerspective(img_np, np.float32(M), (256, 256))
    
    img_np = transform.warp(img_np, tform.inverse)
        
    return img_np

def rotation_transformation(angle, img_np):
    
    img_np = transform.rotate(img_np, angle, resize=False)
    
    if (angle > 45):
        zoom_in_factor = 90-angle
    elif (angle < -45):
        zoom_in_factor = 90-np.abs(angle)
    else:
        zoom_in_factor = np.abs(angle)
        
    M = [[1+np.sin(math.radians(zoom_in_factor)), 0., -128*np.sin(math.radians(zoom_in_factor))],
         [0., 1+np.sin(math.radians(zoom_in_factor)), -128*np.sin(math.radians(zoom_in_factor))],
         [0., 0., 1]]
        
    img_np = cv.warpPerspective(img_np, np.float32(M), (256, 256))
        
    return img_np

def scaling_transformation(factor, img_np):
    M = [[factor, 0., 128*(1-factor)],
             [0., factor, 128*(1-factor)],
             [0., 0., 1.]]

    img_np = cv.warpPerspective(img_np, np.float32(M), (256, 256))
        
    return img_np


def get_fish_xn_yn(source_x, source_y, radius, distortion):
    """
    Get normalized x, y pixel coordinates from the original image and return normalized 
    x, y pixel coordinates in the destination fished image.
    :param distortion: Amount in which to move pixels from/to center.
    As distortion grows, pixels will be moved further from the center, and vice versa.
    """

    if 1 - distortion*(radius**2) == 0:
        return source_x, source_y

    return source_x / (1 - (distortion*(radius**2))), source_y / (1 - (distortion*(radius**2)))


def fish(img, distortion_coefficient):
    """
    :type img: numpy.ndarray
    :param distortion_coefficient: The amount of distortion to apply.
    :return: numpy.ndarray - the image with applied effect.
    """

    # prepare array for dst image
    dstimg = np.zeros_like(img)

    # floats for calcultions
    
    (w, h) = (256, 256)
    
#     (center_x, center_y) = np.random.randint(48, 208, 2)
    (center_x, center_y) = (208, 208)
#     print(center_x, center_y)
    (center_x, center_y) = float((2*center_x-w)/w), float((2*center_y-h)/h)

    
    
    # easier calcultion if we traverse x, y in dst image
    for x in range(len(dstimg)):
        for y in range(len(dstimg[x])):

            # normalize x and y to be in interval of [-1, 1]
            xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)

            # get xn and yn distance from normalized center
            rd = math.sqrt((xnd-center_x)**2 + (ynd-center_y)**2)
                        
            # new normalized pixel coordinates
            xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient)
            
            # convert the normalized distorted xdn and ydn back to image pixels
            xu, yu = int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)

            # if new pixel is in bounds copy from source pixel to destination pixel
            if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
                dstimg[x][y] = img[xu][yu]

    return dstimg.astype(np.uint8)