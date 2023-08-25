import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import label
import denoise_normal
from skimage.morphology import skeletonize_3d

def cal_rang(row, range, top_lim, bot_lim):
    ratio = float(range/660)
    max_range = range - (top_lim * ratio)
    min_range = (row - bot_lim) * ratio
    return max_range, min_range

def coefShift(img1, img2, num):
    if (len(img1) == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if (len(img2) == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    colRange = np.arange((-1) * num, num + 1, 1)
    rowRange = colRange * (-1)
    rowList = []
    coefList = []
    row, col = img1.shape
    for rowInx in rowRange:
        colList = []
        for colInx in colRange:
            trans_matrix = np.float32([[1, 0, colInx], [0, 1, rowInx]])
            imgShift = cv2.warpAffine(img2, trans_matrix, (col, row))

            # ! Quadrant 1
            if rowInx > 0 and colInx > 0:
                imgRef = img1[rowInx:500, colInx:768]
                imgShift = imgShift[rowInx:500, colInx:768]
            # ! Quadrant 2
            elif rowInx > 0 and colInx < 0:
                imgRef = img1[rowInx:500, 0:768 + colInx]
                imgShift = imgShift[rowInx:500, 0:768 + colInx]
            # ! Quadrant 3
            elif rowInx < 0 and colInx < 0:
                imgRef = img1[0:500 + rowInx, 0:768 + colInx]
                imgShift = imgShift[0:500 + rowInx, 0:768 + colInx]
            # ! Quadrant 4
            elif rowInx < 0 and colInx > 0:
                imgRef = img1[0:500 + rowInx, colInx:768]
                imgShift = imgShift[0:500 + rowInx, colInx:768]
            # ! Origin
            elif colInx == 0 and rowInx == 0:
                imgRef = img1[0:row, 0:col]
                imgShift = imgShift[0:row, 0:col]
            # ! row axis
            elif colInx == 0 and rowInx != 0:
                if rowInx > 0:
                    imgRef = img1[rowInx:row, 0:col]
                    imgShift = imgShift[rowInx:row, 0:col]
                elif rowInx < 0:
                    imgRef = img1[0:row+rowInx, 0:col]
                    imgShift = imgShift[0:row+rowInx, 0:col]
            # ! col axis
            elif rowInx == 0 and colInx != 0:
                if colInx > 0:
                    imgRef = img1[0:row, colInx:col]
                    imgShift = imgShift[0:row, colInx:col]
                elif colInx < 0:
                    imgRef = img1[0:row, 0:col+colInx]
                    imgShift = imgShift[0:row, 0:col+colInx]

            coef = cv2.matchTemplate(imgRef, imgShift, cv2.TM_CCOEFF_NORMED)
            colList.append(coef[0][0])
        coefList.append(max(colList))
        rowList.append((rowInx, max(enumerate(colList), key=(lambda x: x[1]))))
    posShift = rowList[np.argmax(coefList)]
    shiftRow = posShift[0]
    shiftCol = posShift[1][0] - num
    print(shiftRow, shiftCol)

    trans_matrix = np.float32([[1, 0, shiftCol], [0, 1, shiftRow]])
    img2 = cv2.warpAffine(img2, trans_matrix, (col, row))
    # res = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    return img2
def shiftImage(im1, im2, im3, im4, im5, num):
    out2 = coefShift(im1, im2, num)
    out3 = coefShift(im1, im3, num)
    out4 = coefShift(im1, im4, num)
    out5 = coefShift(im1, im5, num)

    res = cv2.addWeighted(im1, 0.5, out2, 0.5, 0)
    res = cv2.addWeighted(res, 0.5, out3, 0.5, 0)
    res = cv2.addWeighted(res, 0.5, out4, 0.5, 0)
    res = cv2.addWeighted(res, 0.5, out5, 0.5, 0)

    return res

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
    plt.imshow(labeled)
    plt.show()

    ske = skeletonize_3d(labeled)
    # plt.imshow(ske)
    # plt.show()
    y_max, y_min = 0, row
    for x in range(0, col):
        for y in range(0, row):
            # print(y_max, y_min)
            if (ske[y][x] > 0):
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
    print("angle: ", angle)
    if(5<=angle<=70):
        if(x_ymin>x_ymax):
            mode = "right"
            print("right /")
        elif(x_ymin<x_ymax):
            mode = "left"
            print("left \\")
    elif(75<=angle<=90):
        mode = "straight"
        print("straight |")
    else:
        mode = "straight"
        print("cannot approximate")
    return mode

rootpath = r"C:\Users\mook\PycharmProjects\LSM\images"
scene = 5  # 19
stop_scene = 3100  # 160
r = 20

crop_lim = [25, 515, 10, 758]

start_fig = scene
for i in range(start_fig, stop_scene):
    imgpath = rootpath + "\\RTheta_img_"
    if (i==scene):
        print("FIRST FRAME")
        img_1 = cv2.imread(imgpath + str(i - 4) + ".jpg", 0)
        img_2 = cv2.imread(imgpath + str(i - 3) + ".jpg", 0)
        img_3 = cv2.imread(imgpath + str(i - 2) + ".jpg", 0)
        img_4 = cv2.imread(imgpath + str(i - 1) + ".jpg", 0)
        img_5 = cv2.imread(imgpath + str(i) + ".jpg", 0)
        img_1 = img_1[crop_lim[0]:crop_lim[1], :]
        img_2 = img_2[crop_lim[0]:crop_lim[1], :]
        img_3 = img_3[crop_lim[0]:crop_lim[1], :]
        img_4 = img_4[crop_lim[0]:crop_lim[1], :]
        img_5 = img_5[crop_lim[0]:crop_lim[1], :]
        frame = shiftImage(img_1, img_2, img_3, img_4, img_5, 10)
        # frame = frame[crop_lim[0]:crop_lim[1], :]
        # frame = cv2.imread(rootpath + "\\multilook.png", 0)
        frame = frame[:490, crop_lim[2]: crop_lim[3]]
    else:
        frame = cv2.imread(imgpath + str(i) + ".jpg", 0)
        frame = frame[crop_lim[0]:crop_lim[1], crop_lim[2]: crop_lim[3]]
    img = frame
    row, col = img.shape
    max_range, min_range = cal_rang(660, r, crop_lim[0], crop_lim[1])
    print("max range: %.2f meters, min range: %.2f meters" % (max_range, min_range))

    """PRE-PROCESSING PART"""
    img = cv2.GaussianBlur(img, (9, 9), 3)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_denoise = denoise_normal.denoise_range(img, max_range, min_range)
    img_denoise_sh = img_denoise.astype(dtype=np.uint8)
    mode = direction_est(img_denoise_sh)
    print("mode: ", mode)