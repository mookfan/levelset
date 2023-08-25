import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from ParamsSetting import generalparams, preprocessingparams

def computeCropRange(row, range, top_lim, bot_lim):
    ratio = float(range / row)
    stop_range = range - (top_lim * ratio)
    start_range = (row - bot_lim) * ratio
    return stop_range, start_range

def compensateRange(img, stop_range):
    img_gray = img.astype(np.float32)
    rows, cols = img_gray.shape
    range_resolution = stop_range / rows
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
    return img_denoise

def guassianFiter1D(img):
    gaussian1DWindow = preprocessingparams.gaussian1d_win
    guassian1DKernel = signal.gaussian(gaussian1DWindow, std=int(gaussian1DWindow / (3*2)))
    halfGaussian1DWindow = int(gaussian1DWindow / 2)
    kernel = np.zeros((gaussian1DWindow, gaussian1DWindow))
    """Horizontal"""
    guassian1DHor = kernel
    guassian1DHor[halfGaussian1DWindow:halfGaussian1DWindow + 1, :] = guassian1DKernel
    dstHor = cv2.filter2D(img, -1, guassian1DHor)
    """Vertical"""
    kernel_ver = np.transpose(guassian1DHor)
    dstVert = cv2.filter2D(img, -1, kernel_ver)
    """Left diagonal"""
    kernel_Ldiag = np.diag(guassian1DKernel)
    dstLeftDiag = cv2.filter2D(img, -1, kernel_Ldiag)
    """Right diagonal"""
    kernel_Rdiag = kernel
    for i in range(gaussian1DWindow):
        for j in range(gaussian1DWindow):
            kernel_Rdiag[i][j] = kernel_Ldiag[gaussian1DWindow - i - 1][j]
    dstRightdDiag = cv2.filter2D(img, -1, kernel_Rdiag)
    res = np.zeros(dstHor.shape)
    for rows in range(0, dstHor.shape[0]):
        for cols in range(0, dstHor.shape[1]):
            res[rows][cols] = max(dstHor[rows][cols], dstVert[rows][cols], dstLeftDiag[rows][cols], dstRightdDiag[rows][cols])

    # plt.subplot(221)
    # plt.title("Horizontal")
    # plt.imshow(dstHor, 'gray')
    # plt.subplot(222)
    # plt.title("Vertical")
    # plt.imshow(dstVert, 'gray')
    # plt.subplot(223)
    # plt.title("Left diagonal")
    # plt.imshow(dstLeftDiag, 'gray')
    # plt.subplot(224)
    # plt.title("Right diagonal")
    # plt.imshow(dstRightdDiag, 'gray')
    # plt.show()
    return res

def preProcess(frame, cropFrameLimit, num):
    maxRow = generalparams.max_row
    range = preprocessingparams.range
    gaussianWindow = preprocessingparams.gaussian_win
    gaussianStd = preprocessingparams.gaussian_std
    print("\nPreprocessing of frame %d..." %num)
    print("Gaussian filtering...")
    im_gauss = cv2.GaussianBlur(frame, (gaussianWindow, gaussianWindow), gaussianStd)
    print("Normalize the result of gaussian filter...")
    im_norm = cv2.normalize(im_gauss, None, 0, 255, cv2.NORM_MINMAX)
    print("Compensate range effect of the FLS image...")
    startRange, stopRange = computeCropRange(maxRow, range, cropFrameLimit[0], cropFrameLimit[1])
    print("start range: %.2f meters, stop range: %.2f meters" % (startRange, stopRange))
    im_denoise = compensateRange(im_norm, stopRange)

    im_gauss1D = guassianFiter1D(frame)
    im_gauss1Dnorm = cv2.normalize(im_gauss1D, None, 0, 255, cv2.NORM_MINMAX)
    im_gauss1Ddenoise = compensateRange(im_gauss1Dnorm, stopRange)

    # plt.subplot(121)
    # plt.title("Gaussian 2D")
    # plt.imshow(im_denoise, 'gray')
    # plt.subplot(122)
    # plt.title("Gaussian 1D")
    # plt.imshow(im_gauss1Ddenoise, 'gray')
    # plt.show()
    return im_gauss1Ddenoise