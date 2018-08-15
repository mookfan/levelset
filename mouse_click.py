import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io
from matplotlib.widgets  import RectangleSelector
from scipy import ndimage
import matplotlib.patches as patches
import cv2

x, y = 0, 0 
width, height = 0, 0



def line_select_callback(eclick, erelease):
    global x, y, width, height
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    rect = patches.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
    print rect
    x, y = min(x1,x2),min(y1,y2)
    width, height = np.abs(x1-x2), np.abs(y1-y2)
    ax.add_patch(rect)

def cov(img):
    a = img[348:360, 440:525]
    b = img[400:412, 424:509]
    c = img[300:312, 350:435]
    cv2.imshow("a", a)
    # cv2.imshow("b", b)
    cv2.imshow("c", c)
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    a = np.reshape(a, 1020)
    print a
    b = np.reshape(b, 1020)
    c = np.reshape(c, 1020)
    print "mean_a: ", mean_a
    print "mean_b: ", mean_b
    cov = np.cov(a, c)
    print "cov: ", cov
    return cov

if __name__ == "__main__":
    img = cv2.imread('image_RTheta.jpg',0)
    # # img = cv2.imread('Feature.png', 0)
    # fig, ax = plt.subplots()
    # line = ax.imshow(img, cmap = 'gray')

    # rows, cols = img.shape
    # im = np.zeros((rows, cols), np.uint8)
    # rs = RectangleSelector(ax, line_select_callback,
    #                     drawtype='box', useblit=False, button=[1], 
    #                     minspanx=5, minspany=5, spancoords='pixels', 
    #                     interactive=True)
    # plt.show()
    # # print x, y, width, height
    # x, y, width, height = int(x), int(y), int(width), int(height)
    # crop = img[y:y+height,x:x+width]
    # mean = np.mean(crop)
    # variance = ndimage.variance(crop)
    # print "mean: ", mean
    # print "variance", variance
    cov = cov(img)
    # # print "cov: ", np.cov(crop)
    # cv2.imshow("mask", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()