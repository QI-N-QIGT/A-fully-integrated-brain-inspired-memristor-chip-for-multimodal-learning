import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


def add_gaussian_noise(image, sigma=80):
    '''Permutate the pixels of an image according to [permutation].
    [image]         3D-tensor containing the image
    [noise]   每一个像素点的噪声值
    '''

    img = np.array(image)
    h,w = img.shape
    noise = np.random.normal(0, sigma, (h, w))
    noisy_image = np.clip(img + noise, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image)

    return noisy_image

# 进行blur操作

def add_blur(image, blur_sigma=2):
    '''Permutate the pixels of an image according to [permutation].
    [image]         3D-tensor containing the image
    [blur_sigma]   高斯核的大小
    '''

    img = np.array(image)
    blurred_image = gaussian_filter(img, sigma=(blur_sigma, blur_sigma))
    noisy_image = Image.fromarray(blurred_image)
    return noisy_image

# 进行mask的操作
def occlude_image(image, occlusion_factor = 0.3):
    '''
    # 图像遮挡
    # occlusion_factor: 遮挡区域占图像总面积的比例
    # 返回遮挡前的图像和遮挡后的图像
    '''

    img = np.array(image)
    image_size = img.shape[0]
    occlusion_size = int(occlusion_factor * image_size)
    occlusion_x = random.randint(0, image_size - occlusion_size)
    occlusion_y = random.randint(0, image_size - occlusion_size)

    img[occlusion_x:occlusion_x+occlusion_size, occlusion_y:occlusion_y+occlusion_size] = 0
    occlusion_img = Image.fromarray(img)

    return occlusion_img
#----------------------------------------------------------------------------------------------------------#

# 进行变暗的操作
def dark_image(image, dark_factor = 0.1):
    '''
    # 图像变暗
    # dark_factor: 图像变暗的比例, 越小越暗
    # 返回变暗之后的图像
    '''
    img = np.array(image)
    img = np.clip(img * dark_factor, 0, 255).astype(np.uint8)
    image_dark = Image.fromarray(img)

    return image_dark
