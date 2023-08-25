import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import gamma, norm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import denoise

def distribution(img, pipe, bg, distrib):
    rows, cols = img.shape
    x = np.arange(0, 256, 1)
    img_1D = np.reshape(img, rows * cols)
    print("img_1D type: ", img_1D.dtype)
    img_1D = img_1D.astype(np.uint8)
    if(distrib=="gamma"):
        shape1, loc1, scale1 = gamma.fit(pipe)
        print("pipe")
        print("shape1(k): %.2f, loc1: %.2f, scale1(theta): %.2f" % (shape1, loc1, scale1))
        g1 = gamma.pdf(x=x, a=shape1, loc=loc1, scale=scale1)
        log_g1 = gamma.logpdf(x=x, a=shape1, loc=loc1, scale=scale1)
        logpdf_pipe_1D = log_g1[img_1D]
        logpdf_pipe = np.reshape(logpdf_pipe_1D, (rows, cols))

        shape2, loc2, scale2 = gamma.fit(bg)
        print("background")
        print("shape2(k): %.2f, loc2: %.2f, scale2(theta): %.2f" % (shape2, loc2, scale2))
        g2 = gamma.pdf(x=x, a=shape2, loc=loc2, scale=scale2)
        log_g2 = gamma.logpdf(x=x, a=shape2, loc=loc2, scale=scale2)
        logpdf_bg_1D = (log_g2[img_1D] * (img_1D > 11)) + (log_g2[12] * (img_1D <= 11))
        logpdf_bg = np.reshape(logpdf_bg_1D, (rows, cols))

        ratio = logpdf_bg / logpdf_pipe
        a_infs = np.where(np.isnan(ratio))

    elif(distrib=="gaussian"):
        loc_p, scale_p = norm.fit(pipe)
        print("pipe")
        print("mean: %.2f, std: %.2f" % (loc_p, scale_p))
        g1 = norm.pdf(x=x, loc=loc_p, scale=scale_p)
        log_g1 = norm.logpdf(x=x, loc=loc_p, scale=scale_p)
        logpdf_pipe_1D = log_g1[img_1D]
        logpdf_pipe = np.reshape(logpdf_pipe_1D, (rows, cols))

        loc_bg, scale_bg = norm.fit(bg)
        print("background")
        print("mean: %.2f, std: %.2f" % (loc_bg, scale_bg))
        g2 = norm.pdf(x=x, loc=loc_bg, scale=scale_bg)
        log_g2 = norm.logpdf(x=x, loc=loc_bg, scale=scale_bg)
        logpdf_bg_1D = (log_g2[img_1D] * (img_1D > 11)) + (log_g2[12] * (img_1D <= 11))
        logpdf_bg = np.reshape(logpdf_bg_1D, (rows, cols))
        ratio = logpdf_bg / logpdf_pipe

    # fig = plt.figure()
    # fig.suptitle(distrib)
    # ax = plt.subplot(121)
    # ax.set_title("Pipe")
    # ax.hist(pipe, 256, [0, 256], normed=True)
    # ax.plot(x, g1, label="pdf of pipe", linewidth=2, color='red')
    # ax.plot
    # bx = plt.subplot(122)
    # bx.set_title("Background")
    # bx.hist(bg, 256, [0, 256], normed=True)
    # bx.plot(x, g2, label="pdf of background", linewidth=2, color='orange')
    # plt.show()
    # # plt.subplot(231), plt.imshow(img, 'gray')
    # # plt.title("original_image")
    # # plt.subplot(234), plt.plot(g1, label="pdf of pipe")
    # # plt.subplot(234), plt.plot(g2, 'm', label="pdf of background")
    # # plt.title("pdf")
    # # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # # plt.grid(True)
    # # plt.subplot(236), plt.plot(log_g1, label="logpdf of pipe")
    # # plt.subplot(236), plt.plot(log_g2, 'm', label="logpdf of background", )
    # # plt.title("logpdf")
    # # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # # plt.xlim([0, 256])
    # # plt.grid(True)
    #
    return loc_p, scale_p, loc_bg, scale_bg


def cal_params(img, mask, scence):
    distrib = "gaussian"
    # # img = cv2.imread(r"C:\Users\mook\PycharmProjects\LSM\images_2\RTheta_img_"+str(scene)+".jpg")
    # img = cv2.imread(r"C:\Users\mook\PycharmProjects\LSM\images_2\multilook.png")
    # img = img[25:525, :]
    #
    # """Denoise"""
    # img = denoise.denoise_range(img, 0.0, 30.0)
    # img = img.astype(dtype = np.uint8)

    # mask_ori = cv2.imread(r"C:\Users\mook\PycharmProjects\LSM\mask\mask_RTheta_img_"+str(scene)+".png", 0)
    # mask_ori = cv2.imread(r"C:\Users\mook\PycharmProjects\LSM\experiment\090119\mask_A\14 pixels\RTheta_img_70.png", 0)
    # mask_ori = mask_ori[25:525, :]

    rows, cols = img.shape
    print("total pixel: ", rows*cols)

    mask = mask.astype(np.uint8)
    _, mask_pipe = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow("mask_pipe", mask_pipe)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    _, mask_bg = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("mask_bg", mask_bg)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()

    """Pipe"""
    pipe_masked = cv2.bitwise_and(img,img,mask = mask_pipe)
    # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\mask\pipe_"+str(scene)+".png", pipe_masked)
    # cv2.imshow("pipe mask", pipe_masked)
    pipe_1D = np.reshape(pipe_masked, rows*cols)
    po = np.nonzero(pipe_1D)
    p = pipe_1D[po]
    print ("number of pipe's pixel: ",len(p))
    prob_pipe = len(p)/(rows*cols)

    """bg"""
    bg_masked = cv2.bitwise_and(img,img,mask = mask_bg)
    # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\mask\bg_" + str(scene) + ".png", bg_masked)
    # cv2.imshow("mask_bg", bg_masked)
    bg_1D = np.reshape(bg_masked, rows*cols)
    no = np.nonzero(bg_1D)
    n = bg_1D[no]
    print ("number of bg's pixel: ",len(n))
    prob_bg= len(n)/(rows*cols)

    mean_p, std_p, mean_bg, std_bg = distribution(img, p, n, distrib)

    # plt.subplot(232), plt.imshow(pipe_masked, 'gray')
    # plt.title("pipe")
    # plt.subplot(233), plt.imshow(bg_masked, 'gray')
    # plt.title("background")
    # plt.show()
    # # cv2.waitKey(-1)
    # # cv2.destroyAllWindows()
    #
    # plt.subplot(121)
    # plt.title("pipe_histrogram")
    # plt.hist(p, 256, [0, 256])
    # plt.subplot(122)
    # plt.title("bg_histrogram")
    # plt.hist(n, 256, [0, 256])
    # plt.show()


    return mean_p, std_p, mean_bg, std_bg, prob_pipe, prob_bg


