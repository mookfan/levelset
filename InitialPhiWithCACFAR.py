import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.morphology import skeletonize_3d
from scipy.ndimage.measurements import label

def cacfar (img_gray, box_size, guard_size, pfa):
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
    kernel_beta = (1.0 / n_ref_cell) * kernel_beta
    beta_pow = cv2.filter2D(img_gray, ddepth = cv2.CV_32F, kernel = kernel_beta)
    thres = alpha * (beta_pow)
    out = (255 * (img_gray > thres)) + (0 * (img_gray < thres))
    return out

def phiInitialization(img, ref_size, guard_size, pfa, angleCurved, angleStraight, majorLenght):
    canvas2 = np.zeros(img.shape, np.uint8)
    row, col = img.shape
    caf = cacfar(img, ref_size, guard_size, pfa)
    labeled, num_labeled = label(caf)
    max_num = 0
    max_label = 0
    for i in range(1, num_labeled + 1):
        id = np.where(labeled[:, :] == i)
        num = len(id[0])
        if (num > max_num):
            max_num = num
            max_label = i
    labeled = (0 * (labeled != max_label))+(1 * (labeled == max_label))
    labeled_save = (0 * (labeled == 0))+(255 * (labeled == 1))
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
            if (ske[y][x] > 0):
                point_x.append(x)
                point_y.append(y)
                if (y > y_max):
                    y_max = y
                    x_ymax = x
                if (y < y_min):
                    y_min = y
                    x_ymin = x
    delta_x, delta_y = abs(x_ymax - x_ymin), (y_max - y_min)
    if (delta_x == 0):
        delta_x = 1
    ratio_delta = delta_y / delta_x
    angle = math.degrees(math.atan(ratio_delta))
    canvas = np.zeros(img.shape, np.float32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if(angleCurved[0] <= angle <= angleCurved[1]):
        canvas2 = labeled.copy()
        canvas2 = (255 * (canvas2 == 1)) + (0 * (canvas2 == 0))
        canvas2 = canvas2.astype(np.uint8)
        contours, _ = cv2.findContours(canvas2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        canvas2 = cv2.cvtColor(canvas2, cv2.COLOR_GRAY2BGR)
        MA = 0
        for i in contours:
            if(len(i)>= 5):
                ellipse = cv2.fitEllipse(i)
                # print("ellipse: ", ellipse)
                cv2.ellipse(canvas2, ellipse, (0, 255, 0), 2)
                if(ellipse[1][0] > MA):
                    MA = ellipse[1][0]
        # print("Major axis: %.2f" %(ellipse[1][0]))
        # plt.imshow(canvas2)
        # plt.show()
        if(MA > majorLenght):
            if (y_max < row):
                for i in range(y_max, row):
                    point_x.append(x_ymax)
                    point_y.append(i)
            if(x_ymin > x_ymax):
                shape = "right"
                print("mode: right /")
                cv2.putText(labeled_save, 'right', (5, 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                for k in range(0, len(point_x)):
                    for i in range(0, col):
                        for j in range(0, row):
                            if (i >= point_x[k] and j >= point_y[k]):
                                canvas[j][i] = 1
            elif(x_ymin < x_ymax):
                shape = "left"
                print("mode: left \\")
                cv2.putText(labeled_save, 'left', (5, 20), font, 1, (96, 186, 255), 1, cv2.LINE_AA)
                for k in range(0, len(point_x)):
                    for i in range(0, col):
                        for j in range(0, row):
                            if (i <= point_x[k] and j >= point_y[k]):
                                canvas[j][i] = 1
        else:
            print("mode: Defalt (straight [cannot pass through curve condition])")
            shape = "straight"
            x_pos = int((col) / 2)
            canvas[0: row, x_pos: col] = 1

    elif(angleStraight[0] <= angle <= angleStraight[1]):
        shape = "straight"
        print("mode: straight |")
        cv2.putText(labeled_save, 'straight', (5, 20), font, 1, (211, 229, 149), 1, cv2.LINE_AA)
        x_pos = int((x_ymin + x_ymax) / 2)
        canvas[0: row, x_pos: col] = 1
    else:
        print("mode: Defalt ([unpass angle condition])")
        shape = "straight"
        x_pos = int((col) / 2)
        canvas[0: row, x_pos: col] = 1
    canvas = (1 * (canvas == 1)) + (-1 * (canvas == 0))

    ske_save = (255 * (ske > 0)) + (0 * (ske <= 0))
    ske_save = ske_save.astype(np.uint8)
    # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\ske.png", ske_save)
    # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\label.png", labeled_save)
    canvas_save = (255 * (canvas == 1)) + (0 * (canvas == -1))
    canvas_save = canvas_save.astype(np.uint8)
    # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\init.png", canvas_save)

    canv = [canvas, canvas2]
    return shape, labeled_save, canv