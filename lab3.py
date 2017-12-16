#encoding: utf-8

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import random
import cv2
import scipy.misc
import scipy.signal
import scipy.ndimage

def medium_filter(im, x, y, step):
    sum_s=[]
    for k in range(-int(step/2),int(step/2)+1):
        for m in range(-int(step/2),int(step/2)+1):
            sum_s.append(im[x+k][y+m])
    sum_s.sort()
    return sum_s[(int(step*step/2)+1)]

def mean_filter(im, x, y, step):
    sum_s = 0
    for k in range(-int(step/2),int(step/2)+1):
        for m in range(-int(step/2),int(step/2)+1):
            sum_s += im[x+k][y+m] / (step*step)
    return sum_s

def convert_2d(r):
    n = 3
    # 3*3 滤波器, 每个系数都是 1/9
    window = np.ones((n, n)) / n ** 2
    # 使用滤波器卷积图像
    # mode = same 表示输出尺寸等于输入尺寸
    # boundary 表示采用对称边界条件处理图像边缘
    s = scipy.signal.convolve2d(r, window, mode='same', boundary='symm')
    return s.astype(np.uint8)

# def convert_3d(r):
#     s_dsplit = []
#     for d in range(r.shape[2]):
#         rr = r[:, :, d]
#         ss = convert_2d(rr)
#         s_dsplit.append(ss)
#     s = np.dstack(s_dsplit)
#     return s


def add_salt_noise(img):
    rows, cols, dims = img.shape 
    R = np.mat(img[:, :, 0])
    G = np.mat(img[:, :, 1])
    B = np.mat(img[:, :, 2])

    Grey_sp = R * 0.299 + G * 0.587 + B * 0.114
    Grey_gs = R * 0.299 + G * 0.587 + B * 0.114

    snr = 0.9
    mu = 0
    sigma = 0.12
    
    noise_num = int((1 - snr) * rows * cols)

    for i in range(noise_num):
        rand_x = random.randint(0, rows - 1)
        rand_y = random.randint(0, cols - 1)
        if random.randint(0, 1) == 0:
            Grey_sp[rand_x, rand_y] = 0
        else:
            Grey_sp[rand_x, rand_y] = 255
    
    Grey_gs = Grey_gs + np.random.normal(0, 48, Grey_gs.shape)
    Grey_gs = Grey_gs - np.full(Grey_gs.shape, np.min(Grey_gs))
    Grey_gs = Grey_gs * 255 / np.max(Grey_gs)
    Grey_gs = Grey_gs.astype(np.uint8)

    # 中值滤波
    Grey_sp_mf = scipy.ndimage.median_filter(Grey_sp, (8, 8))
    Grey_gs_mf = scipy.ndimage.median_filter(Grey_gs, (8, 8))

    # 均值滤波
    n = 3
    window = np.ones((n, n)) / n ** 2
    Grey_sp_me = convert_2d(Grey_sp)
    Grey_gs_me = convert_2d(Grey_gs)

    plt.subplot(321)
    plt.title('Grey salt and pepper noise')
    plt.imshow(Grey_sp, cmap='gray')
    plt.subplot(322)
    plt.title('Grey gauss noise')
    plt.imshow(Grey_gs, cmap='gray')

    plt.subplot(323)
    plt.title('Grey salt and pepper noise (medium)')
    plt.imshow(Grey_sp_mf, cmap='gray')
    plt.subplot(324)
    plt.title('Grey gauss noise (medium)')
    plt.imshow(Grey_gs_mf, cmap='gray')

    plt.subplot(325)
    plt.title('Grey salt and pepper noise (mean)')
    plt.imshow(Grey_sp_me, cmap='gray')
    plt.subplot(326)
    plt.title('Grey gauss noise (mean)')
    plt.imshow(Grey_gs_me, cmap='gray')
    plt.show()

    


def main():
    img = np.array(Image.open('LenaRGB.bmp'))
    add_salt_noise(img)



if __name__ == '__main__':
    main()