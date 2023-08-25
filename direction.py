import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.morphology import skeletonize_3d
from scipy.ndimage.measurements import label

def cafar (img_gray, box_size, guard_size, pfa):
    n_ref_cell = (box_size * box_size) - (guard_size * guard_size)
    # print ("number of ref cell: %.2f" %(n_ref_cell))

    alpha = n_ref_cell * (math.pow(pfa, (-1.0/n_ref_cell)) - 1)
    # print ("alpha: %.2f" %(alpha))
    # img = cv2.imread(r"D:\Mook\LevelSet\Images\PMap.png")
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """create kernel"""
    kernel_beta = np.ones([box_size, box_size], 'float32')
    width = round((box_size - guard_size) / 2.0)
    height = round((box_size - guard_size) / 2.0)
    kernel_beta[width: box_size - width, height: box_size - height] = 0.0
    # print (kernel_beta)
    kernel_beta = (1.0 / n_ref_cell) * kernel_beta

    beta_pow = cv2.filter2D(img_gray, ddepth = cv2.CV_32F, kernel = kernel_beta)
    # plt.imshow(beta_pow)
    # plt.show()
    #
    # print ("beta_pow(max): %.2f, beta_pow(min): %.2f" %(beta_pow.max(), beta_pow.min()))
    # plt.imshow(beta_pow)
    # plt.show()

    thres = alpha * (beta_pow)
    # print("thres: ", thres)
    # plt.imshow(thres)
    # plt.show()

    out = (255 * (img_gray > thres)) + (0 * (img_gray < thres))
    return out

def direction_est(img):
    row, col = img.shape
    caf = cafar(img, 51, 41, 0.3)
    # plt.imshow(caf)
    # plt.show()
    labeled, num_labeled = label(caf)
    max_num = 0
    max_label = 0
    for i in range(1, num_labeled + 1):
        id = np.where(labeled[:, :] == i)
        num = len(id[0])
        # print("i: ", i)
        # print("num: ", num)
        if (num > max_num):
            max_num = num
            max_label = i
    labeled = (0*(labeled!=max_label))+(1*(labeled==max_label))

    labeled_save = (0*(labeled==0))+(255*(labeled==1))
    labeled_save = labeled_save.astype(np.uint8)
    # plt.imshow(labeled_save)
    # plt.show()

    ske = skeletonize_3d(labeled)
    # plt.imshow(ske)
    # plt.show()
    point_x, point_y = [], []
    y_max, y_min = 0, row
    for x in range(0, col):
        for y in range(0, row):
            # print(y_max, y_min)
            if (ske[y][x] > 0):
                point_x.append(x)
                point_y.append(y)
                # print(x, y)
                if (y > y_max):
                    y_max = y
                    x_ymax = x
                if (y < y_min):
                    y_min = y
                    x_ymin = x
    print("max x, y: ", x_ymax, y_max)
    print("min x, y: ", x_ymin, y_min)
    delta_x, delta_y = abs(x_ymax - x_ymin), (y_max - y_min)
    if (delta_x == 0):
        delta_x = 1
    ratio_delta = delta_y / delta_x
    angle = math.degrees(math.atan(ratio_delta))

    canvas = np.zeros(img.shape, np.float32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    print("angle: ", angle)
    if(5<=angle<=70):
        """added"""
        if (y_max < row):
            for i in range(y_max, row):
                point_x.append(x_ymax)
                point_y.append(i)
        if(x_ymin>x_ymax):
            mode = "right"
            print("right /")
            cv2.putText(labeled_save, 'right', (5, 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            for k in range(0, len(point_x)):
                for i in range(0, col):
                    for j in range(0, row):
                        if (i >= point_x[k] and j >= point_y[k]):
                            canvas[j][i] = 1
        elif(x_ymin<x_ymax):
            mode = "left"
            print("left \\")
            cv2.putText(labeled_save, 'left', (5, 20), font, 1, (96, 186, 255), 1, cv2.LINE_AA)
            for k in range(0, len(point_x)):
                for i in range(0, col):
                    for j in range(0, row):
                        if (i <= point_x[k] and j >= point_y[k]):
                            canvas[j][i] = 1
    elif(75<=angle<=90):
        mode = "straight"
        print("straight |")
        cv2.putText(labeled_save, 'straight', (5, 20), font, 1, (211, 229, 149), 1, cv2.LINE_AA)
        x_pos = int((x_ymin + x_ymax) / 2)
        canvas[0: row, x_pos: col] = 1
    else:
        mode = "straight"
        print("cannot approximate")
        x_pos = int((x_ymin + x_ymax) / 2)
    canvas = (1*(canvas==1)) + (-1*(canvas==0))
    return mode, labeled_save, canvas