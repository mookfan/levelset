import cv2
import numpy as np
import denoise_normal
import levelset_weighted
import matplotlib.pyplot as plt

rootpath = r"C:\Users\mook\PycharmProjects\LSM\images\RTheta_img_"

for k in range (10, 121):
    print("image: #", k)
    imgpath = rootpath+str(k)+".jpg"
    """DENOISE Parameters"""
    start_range = 0.0
    stop_range = 30.0

    """_____IMPORT IMAGE_____"""
    img = cv2.imread(imgpath)
    img = img[25:525, 20:748]  # Crop
    img_filt = cv2.GaussianBlur(img, (9, 9), 3)
    img_filt = cv2.normalize(img_filt, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow("img", img)
    """______DENOISE______"""
    img_denoise = denoise_normal.denoise_range(img_filt, start_range, stop_range)
    img_denoise_sh = img_denoise.astype(dtype=np.uint8)

    # img_denoise_sh =  cv2.cvtColor(img_filt, cv2.COLOR_BGR2GRAY)

    all_size = 45#61
    win_size = 27#31
    guard_size = int((all_size - win_size)/2)

    half_win = int(win_size/2)


    # cv2.imshow("original", img_denoise_sh)

    div_mod = divmod(all_size, 2)
    center = div_mod[0]
    # """try ZERO-PADDING"""
    # padding_img = cv2.copyMakeBorder(img_denoise_sh, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)


    rows, cols = img_denoise_sh.shape


    mask = np.zeros([win_size, win_size], np.float32)
    guard_coeff = 1./(pow(all_size, 2) - pow(win_size, 2))
    guard_kernel = guard_coeff * cv2.copyMakeBorder(mask, guard_size, guard_size, guard_size, guard_size, cv2.BORDER_CONSTANT, value=1)
    guard_kernel[all_size-guard_size:all_size, 0: all_size] = 0

    win_coeff = 1./pow(win_size, 2)
    window_kernel =  win_coeff * np.ones([win_size, win_size], np.float32)


    win_mean = cv2.filter2D(img_denoise,  ddepth = cv2.CV_32F, kernel = window_kernel)
    guard_mean = cv2.filter2D(img_denoise,  ddepth = cv2.CV_32F, kernel = guard_kernel)

    ratio = win_mean/guard_mean

    thres = 1.8 #1.25
    out_ = (img_denoise_sh * (ratio>=thres)) + (0 * (ratio<thres))
    cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\suppress\shadow_win\\" + str(thres) + "_frame_" + str(k) + "_suppress.png",out_)
    out = (ratio * (ratio >= thres)) + (-1 * ratio * (ratio < thres))

    # size = 3
    # morp_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    # out2 = cv2.dilate(out, morp_kernel, iterations=3)

    # cv2.imshow("dilate"+str(k), out)
    # cv2.imshow("suppress"+str(k), out)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()


    # plt.subplot(221)
    # plt.imshow(img)
    #
    # plt.subplot(222)
    # plt.imshow(out)
    #
    # plt.subplot(223)
    # # plt.imshow(win_mean)
    # plt.imshow(window_kernel)
    #
    # plt.subplot(224)
    # # plt.imshow(guard_mean)
    # plt.imshow(guard_kernel)
    #
    # # plt.subplot(121)
    # # plt.imshow(win_mean/guard_mean)
    # #
    # # plt.subplot(122)
    # # plt.imshow(out_)
    # plt.show()