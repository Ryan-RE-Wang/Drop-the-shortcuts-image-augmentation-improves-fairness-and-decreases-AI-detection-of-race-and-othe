import math
import numpy as np
from skimage import transform
import cv2 as cv
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
import nibabel as nib


def crop_image_test(image):
    
    mask=image==0
    if(mask.all()==1):
        return -1
    coords = np.array(np.nonzero(~mask))
    
    top_left = np.min(coords, axis=1)
    
    
    botton_right = np.max(coords, axis=1)
    
    croped_image=image[top_left[0]:botton_right[0], 
                       top_left[1]:botton_right[1]]
    
    return top_left, botton_right

def crop(test):   
    top=1000
    down=0
    l=1000
    r=0

    for i in range(test.shape[0]):
        data=crop_image_test(test[i,:,:])
        if(data==-1):
            continue
            
        if top>data[0][0]:
            top=data[0][0]
        if down<data[1][0]:
            down=data[1][0]
        if l>data[0][1]:
            l=data[0][1]
        if r<data[1][1]:
            r=data[1][1]
    new=[]
    for i in range(test.shape[0]):
        t=test[i,:,:][top:down,l:r]
        if (np.all(t==0)):
            continue
        new.append(t)
    
    new = np.array(new)
    
    return new

def mask_img(img, mask):
    for j in range(mask.shape[0]):
        idx=(mask[j, :, :]==0)
        img[j, :, :][idx] = 0
        
    for j in range(mask.shape[1]):
        idx=(mask[:, j, :]==0)
        img[:, j, :][idx] = 0
        
    for j in range(mask.shape[2]):
        idx=(mask[:, :, j]==0)
        img[:, :, j][idx] = 0
        
    return img
        
def load_img(file_path):
    data = nib.load(file_path)
    data = np.array(data.dataobj)
    return data

def plot(img):
    for i in range(img.shape[0]):
        plt.subplot(16, 16, i+1)
        fig = plt.imshow(img[i, :, :])

        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.gcf().set_size_inches(15,15)
    plt.show()

def shear(img, coef):
    img_np = np.copy(img)
        
    seed = np.random.uniform(-np.pi/coef, np.pi/coef)
    
    for i in range(img.shape[0]):
            img_np[i, :, :] = shear_transform(seed, img_np[i, :, :])

    for i in range(img.shape[1]):
            img_np[:, i, :] = shear_transform(seed, img_np[:, i, :])

    for i in range(img.shape[2]):
            img_np[:, :, i] = shear_transform(seed, img_np[:, :, i])

    img_np = crop(img_np)
    
    return img_np

def scale(img, coef):
    img_np = np.copy(img)
    
    seed = np.random.uniform(coef, 1)
    
    for i in range(img.shape[0]):
            img_np[i, :, :] = scaling_transformation(seed, img_np[i, :, :])

    for i in range(img.shape[1]):
            img_np[:, i, :] = scaling_transformation(seed, img_np[:, i, :])

    for i in range(img.shape[2]):
            img_np[:, :, i] = scaling_transformation(seed, img_np[:, :, i])
    
    return img_np

def rotation(img, coef):
    img_np = np.copy(img)
    
    angle = np.random.uniform(-coef, coef)
    
    for i in range(img.shape[0]):
            img_np[i, :, :] = rotation_transformation(angle, img_np[i, :, :])

    angle = np.random.uniform(-5, 5)
    for i in range(img.shape[1]):
            img_np[:, i, :] = rotation_transformation(angle, img_np[:, i, :])

    angle = np.random.uniform(-5, 5)
    for i in range(img.shape[2]):
            img_np[:, :, i] = rotation_transformation(angle, img_np[:, :, i])

    img_np = crop(img_np)
    
    return img_np

def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)

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
    
    img_np = cv.warpPerspective(img_np, np.float32(M), img_np.shape)
    
    img_np = transform.warp(img_np, tform.inverse)
        
    return img_np

def rotation_transformation(angle, img_np):
    
    img_np = transform.rotate(img_np, angle, resize=False)
        
    return img_np

def scaling_transformation(factor, img_np):
    M = [[factor, 0., 128*(1-factor)],
             [0., factor, 128*(1-factor)],
             [0., 0., 1.]]

    img_np = cv.warpPerspective(img_np, np.float32(M), img_np.shape)
        
    return img_np


def get_fish_zn_xn_yn(source_z, source_x, source_y, radius, distortion):
    """
    Get normalized x, y pixel coordinates from the original image and return normalized 
    x, y pixel coordinates in the destination fished image.
    :param distortion: Amount in which to move pixels from/to center.
    As distortion grows, pixels will be moved further from the center, and vice versa.
    """

    if 1 - distortion*(radius**2) == 0:
        return source_z, source_x, source_y

    return source_z / (1 - (distortion*(radius**2))), source_x / (1 - (distortion*(radius**2))), source_y / (1 - (distortion*(radius**2)))


def fish(img, distortion_coefficient):
    """
    :type img: numpy.ndarray
    :param distortion_coefficient: The amount of distortion to apply.
    :return: numpy.ndarray - the image with applied effect.
    """

    # prepare array for dst image
    dstimg = np.zeros_like(img)

    # floats for calcultions
    
    (c, w, h) = img.shape
    
    (center_z, center_x, center_y) = np.random.randint(48, 208, 3)
    (center_z, center_x, center_y) = float((2*center_z-c)/c), float((2*center_x-w)/w), float((2*center_y-h)/h)

    
    
    # easier calcultion if we traverse x, y in dst image
    for z in range(c):
        for x in range(w):
            for y in range(h):

                # normalize x and y to be in interval of [-1, 1]
                znd, xnd, ynd = float((2*z - c)/c), float((2*x - w)/w), float((2*y - h)/h)

                # get xn and yn distance from normalized center
                rd = math.sqrt((znd - center_z)**2 + (xnd - center_x)**2 + (ynd - center_y)**2)

                # new normalized pixel coordinates
                zdu, xdu, ydu = get_fish_zn_xn_yn(znd, xnd, ynd, rd, distortion_coefficient)

                # convert the normalized distorted xdn and ydn back to image pixels
                zu, xu, yu = int(((zdu + 1)*c)/2), int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)

                # if new pixel is in bounds copy from source pixel to destination pixel
                if 0 <= zu and zu < img.shape[0] and 0 <= xu and xu < img.shape[1] and 0 <= yu and yu < img.shape[2]:
                    dstimg[z][x][y] = img[zu][xu][yu]
                    
    dstimg = crop(dstimg)
      
    return dstimg