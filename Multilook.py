import cv2
import numpy as np
import matplotlib.pyplot as plt
from ParamsSetting import multilookparams

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
    trans_matrix = np.float32([[1, 0, shiftCol], [0, 1, shiftRow]])
    img2 = cv2.warpAffine(img2, trans_matrix, (col, row))
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

def multilookProcess(framePath, cropFrameLimit, num):
    shiftTime = multilookparams.shift_time
    print("\nMultilook of frame %d..." %num)
    # print("Import 5 adjacent frames... (#%d, #%d, #%d, #%d, #%d)" % (num - 4, num - 3, num - 2, num - 1, num))
    # img_1 = cv2.imread(framePath + str(num - 4) + ".png", 0)
    # img_2 = cv2.imread(framePath + str(num - 3) + ".png", 0)
    # img_3 = cv2.imread(framePath + str(num - 2) + ".png", 0)
    # img_4 = cv2.imread(framePath + str(num - 1) + ".png", 0)
    # img_5 = cv2.imread(framePath + str(num) + ".png", 0)
    print("Import 5 adjacent frames... (#%d, #%d, #%d, #%d, #%d)" % (num, num + 1, num + 2, num + 3, num + 4))
    img_1 = cv2.imread(framePath + str(num) + ".png", 0)
    img_2 = cv2.imread(framePath + str(num + 1) + ".png", 0)
    img_3 = cv2.imread(framePath + str(num + 2) + ".png", 0)
    img_4 = cv2.imread(framePath + str(num + 3) + ".png", 0)
    img_5 = cv2.imread(framePath + str(num + 4) + ".png", 0)
    # print("Crop ROI of each adjacent frames...")
    # img_1 = img_1[cropFrameLimit[0]:cropFrameLimit[1], :]
    # img_2 = img_2[cropFrameLimit[0]:cropFrameLimit[1], :]
    # img_3 = img_3[cropFrameLimit[0]:cropFrameLimit[1], :]
    # img_4 = img_4[cropFrameLimit[0]:cropFrameLimit[1], :]
    # img_5 = img_5[cropFrameLimit[0]:cropFrameLimit[1], :]
    print("Process the multilook method using correlation coefficient...")
    res = shiftImage(img_1, img_2, img_3, img_4, img_5, shiftTime)
    print("Crop ROI of the multilook result...")
    res = res[cropFrameLimit[0]:cropFrameLimit[1], cropFrameLimit[2]: cropFrameLimit[3]]
    return res