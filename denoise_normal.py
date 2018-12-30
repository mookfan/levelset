import cv2
# import math
import numpy as np
# import tensorflow as tf
# import scipy.ndimage
# import matplotlib.pyplot as plt


def denoise_range (img, start_range, stop_range):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.float32)

    rows, cols = img_gray.shape
    range_resolution = stop_range / rows

    # img_blur = cv2.GaussianBlur(img_gray, (9, 9), 2)

    map = np.arange(0, rows, 1)
    map = map.reshape((-1, 1))
    map = np.hstack([map] * cols)
    r = stop_range - (range_resolution * map)
    r_2 = r ** 2
    i_adj = r_2 * img_gray
    cv2.normalize(i_adj, i_adj, 0, 255, cv2.NORM_MINMAX)

    i_adj = i_adj.astype(np.float32)
    img_denoise = cv2.add(img_gray, i_adj)

    cv2.normalize(img_denoise, img_denoise, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite(r"D:\Mook\LevelSet\Images\img_denoise.png", img_denoise)

    return img_denoise



