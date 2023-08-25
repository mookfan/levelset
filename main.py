import cv2
import numpy as np
import math
import time
import sys
import pickle as pl
import denoise_normal
import levelset_weighted
import post_process
import distribution_params
import phi_estimate
import direction
import matplotlib.pyplot as plt



def cal_rang(row, range, top_lim, bot_lim):
    ratio = float(range/660)
    stop_range = range - (top_lim * ratio)
    start_range = (row - bot_lim) * ratio
    return stop_range, start_range

"""MULTI-LOOK"""
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

# def cafar (img_gray, box_size, guard_size, pfa):
#     n_ref_cell = (box_size * box_size) - (guard_size * guard_size)
#     alpha = n_ref_cell * (math.pow(pfa, (-1.0/n_ref_cell)) - 1)
#     kernel_beta = np.ones([box_size, box_size], 'float32')
#     width = round((box_size - guard_size) / 2.0)
#     height = round((box_size - guard_size) / 2.0)
#     kernel_beta[width: box_size - width, height: box_size - height] = 0.0
#     kernel_beta = (1.0 / n_ref_cell) * kernel_beta
#     plt.imshow(kernel_beta)
#     plt.show()
#     beta_pow = cv2.filter2D(img_gray, ddepth = cv2.CV_32F, kernel = kernel_beta)
#     thres = alpha * (beta_pow)
#     out = (255 * (img_gray > thres)) + (0 * (img_gray < thres))
#     return out

def process ():
    # mode = input("Mode? (debug, test) ")
    start = time.time()

    mode_in = "debug"
    if(mode_in=="test"):
        rootpath = input("What's path of this dataset? ")
        dataset = input("What's dataset number? ")
        r = int(input("What's range? "))
        crop_lim = [int(x) for x in input("What's limitation row for cropping (top bottom)? ").split()]

        if (dataset == "1"):
            # mean_pipe_pre, std_pipe_pre = 115.69, 43.79
            # mean_bg_pre, std_bg_pre = 72.22, 27.44
            # prob_p_pre, prob_bg_pre  = 13797. / 364000, 350200. / 364000
            scene = 15
            stop_scene = 160

        elif (dataset == "2"):
            # mean_pipe_pre, std_pipe_pre = 99.86, 29.91
            # mean_bg_pre, std_bg_pre = 69.13, 24.47
            # prob_p_pre, prob_bg_pre = 13797. / 364000, 355637. / 364000
            scene = 70
            stop_scene = 1000

    elif(mode_in=="debug"):
        print("You use dataset 1")
        rootpath = r"C:\Users\mook\PycharmProjects\LSM\images"
        scene = 539  # 19
        stop_scene = 740  # 160
        r = 30

        # print("You use dataset Apr_26_2017_213129_2040-2340_4")
        # rootpath = r"C:\Users\mook\PycharmProjects\LSM\dataset\Apr_26_2017_213129_2040-2340_4"
        # scene = 5  # 19
        # stop_scene = 3100  # 160
        # r = 20

        # rootpath = r"C:\Users\mook\PycharmProjects\LSM\dataset\Apr_25_2017_234905_4"
        # scene = 5  # 19
        # stop_scene = 3100  # 160
        # r = 20

        crop_lim = [25, 515, 10, 758]
        # mean_pipe_pre, std_pipe_pre = 115.69, 43.79
        # mean_bg_pre, std_bg_pre = 72.22, 27.44
        # prob_p_pre, prob_bg_pre = 13797. / 364000, 350200. / 364000

    else:
        print("wrong input mode!")
        sys.exit()


    print("1st frame: %d, last frame: %d" %(scene, stop_scene))

    """initial boolean"""
    initial = False #when level set cannot detect our object at all
    re_phi = False

    figure = 368
    start_fig = scene
    for i in range(start_fig, stop_scene):
        """READ IMAGE PART"""
        if (i == scene) or (initial):
            start_bool = True
        else:
            start_bool = False

        imgpath = rootpath + "\\RTheta_img_"
        if(start_bool):
            print("FIRST FRAME")
            img_1 = cv2.imread(imgpath + str(i-4) + ".jpg", 0)
            img_2 = cv2.imread(imgpath + str(i-3) + ".jpg", 0)
            img_3 = cv2.imread(imgpath + str(i-2) + ".jpg", 0)
            img_4 = cv2.imread(imgpath + str(i-1) + ".jpg", 0)
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
        start_range, stop_range = cal_rang(660, r, crop_lim[0], crop_lim[1])
        print("start range: %.2f meters, stop range: %.2f meters" % (start_range, stop_range))

        """PRE-PROCESSING PART"""
        img = cv2.GaussianBlur(img, (9, 9), 3)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_denoise = denoise_normal.denoise_range(img, start_range, stop_range)
        img_denoise_sh = img_denoise.astype(dtype=np.uint8)

        """SET PARAMETERS"""
        if(start_bool) or re_phi:
            mode, cafar,_ = direction.direction_est(img_denoise_sh)
            cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\cafar_direction"+str(i)+".png", cafar)
            iteration = 2600 #2600
            # mean_pipe, std_pipe = mean_pipe_pre, std_pipe_pre
            # mean_bg, std_bg = mean_bg_pre, std_bg_pre
            # prob_p, prob_bg = prob_p_pre, prob_bg_pre
            # eta = 0.25 * np.log10(prob_bg / prob_p)
            # gauss_param = [[mean_pipe, std_pipe], [mean_bg, std_bg]]
            phi_coef = 2.0  # 6.0
            init_phi = (-1.0 * phi_coef) * np.ones(img_denoise_sh.shape, 'float32')
            # init_phi[150: 350, 280:480] = phi_coef

            if(mode == "left"):
                print("""\\""")
            # """\\"""
                iteration = 2600  # 1600
                # pts = np.array([[0, 0], [0, row], [350, row]], np.int32)
                pts = np.array([[0, 200], [0, row], [350, row], [350, 200]], np.int32)
                mew, lamda, v = 0.2, 105.0, 5.5  # -5.0 #initial inside: -
                alpha = 2.0  # 5.0,  0.08 for n sized
                epsilon, t = 2.0, 1.0  # 4.0

            elif(mode == "right"):
                print("""/""")
            # """/"""
            #     iteration = 2600  # 1600
                pts = np.array([[col, 50], [col, row], [600, row]], np.int32)  # triangle
                mew, lamda, v = 0.1, 105.0, 5.5  # -5.0 #initial inside: -
                alpha = 2.0  # 5.0,  0.08 for n sized
                epsilon, t = 2.0, 2.0  # 4.0

            else:
                print("""|""")
            # """|"""
                iteration = 600 # 1600
                pts = np.array([[450, 0], [450, row],[col, row], [col, 0]], np.int32)  # rectangle
                mew, lamda, v = 0.2, 15.0, 1.0  # 70, 5
                alpha = 2.0  # 5.0,  0.08 for n sized
                epsilon, t = 2.0, 1.0  # 4.0

            print("initial")
            # # pts = np.array([[col, 150], [col, row], [550, row]], np.int32)  # triangle
            pts = pts.reshape((-1, 1, 2))
            cv2.fillConvexPoly(init_phi, pts, 2)
            # plt.imshow(init_phi)
            # plt.show()


            levelset_param = [init_phi, iteration, mew, lamda, v, alpha, epsilon, t]

        else:
            # if(i==figure):
            #     """Debug"""
            #     mode = direction.direction_est(img_denoise_sh)
            #     mean_pipe, std_pipe = mean_pipe_pre, std_pipe_pre
            #     mean_bg, std_bg = mean_bg_pre, std_bg_pre
            #     prob_p, prob_bg = prob_p_pre, prob_bg_pre
            #     eta = 0.25 * np.log10(prob_bg / prob_p)
            #     gauss_param = [[mean_pipe, std_pipe], [mean_bg, std_bg]]
            #     phi_coef = 2.0  # 6.0
            #
            #     # plt.title("phi")
            #     phi_added = cv2.imread(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\phi_"+str(figure-1)+".png", 0)
            #     phi_added = (2*(phi_added>0))+(-2*(phi_added<=0))
            #     plt.imshow(phi_added)
            #     plt.show()

            init_phi = phi_added
            iteration = 200
            mew, lamda, v = 0.2, 15.0, -1.0  # -5.0 #initial inside: -
            alpha = 1.0  # 0.5,  0.08 for n sized
            epsilon, t = 1.0, 1.0 # 4.0
            levelset_param = [init_phi, iteration, mew, lamda, v, alpha, epsilon, t]

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("FRAME ", i)
        """LEVEL SET METHOD"""
        phi = levelset_weighted.levelset_cal(img_denoise, levelset_param, i)
        phi_save = (255 * (phi >= 0)) + (0 * (phi) < 0)
        phi_save = phi_save.astype(np.uint8)
        cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\phi_before" + str(i) + ".png", phi_save)


        end = time.time()
        print("Processing time of level set: %.3f sec" % ((end - start)/60))

        # """POST-PROCESSING"""
        # out_thresh = (0 * (phi <= 0)) + (255 * (phi > 0))
        # plt.imshow(out_thresh)
        # plt.show()
        # poly_params_pre = [poly_function_pre]
        # res, canvas, labeled, poly_params, re_phi = post_process.join(out_thresh, i, poly_params_pre)
        # poly_function = poly_params[0]
        #
        # phi_added = ((-1.0 * phi_coef) * (canvas == 0)) + (phi_coef * (canvas > 0))
        # plt.title("result" + str(i))
        # plt.imshow(phi_added)
        # """show in r & theta axis"""
        # rows, cols = img_denoise_sh.shape
        # x, y = np.arange(0, cols + 1, 76.8), np.arange(0, rows + 1, 50.0)
        # x_new, y_new = np.arange(-65, 66, 13.0), np.arange(30, 7, -2.0)
        # fig = plt.figure()
        # plt.xticks(x, x_new, rotation='vertical')
        # plt.yticks(y, y_new)
        # plt.xlabel('theta(degree)', fontsize=15)
        # plt.ylabel('range(meters)', fontsize=15)
        # # plt.savefig(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\frame_" + str(i) + "_result.png")
        # pl.dump(fig, open(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\label_" + str(i) + ".pickle",'wb'))
        # # plt.show()
        # plt.close()
        #
        # phi_save = (255 * (phi_added > 0))+(0 * (phi_added <= 0))
        # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\frame_" + str(i) + "_poly.png", phi_save)
        #
        # phi_save = phi_save.astype(dtype = np.uint8)
        # comp = cv2.addWeighted(img_denoise_sh, 0.7, phi_save, 0.3, 0)
        # fig = plt.figure()
        # plt.title("compare result with original image" + str(i))
        # plt.imshow(comp)
        # plt.savefig(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\frame_" + str(i) + "_result.png")
        # pl.dump(fig, open(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\result_" + str(i) + ".pickle", 'wb'))
        # plt.close()

        phi_update, re_phi = phi_estimate.phi_est(phi, mode)
        phi_added = ((-1.0 * phi_coef) * (phi_update <= 0)) + (phi_coef * (phi_update > 0))

        phi_save = (255 * (phi_added >= 0)) + (0 * (phi_added) < 0)
        phi_save = phi_save.astype(np.uint8)
        cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\phi_" + str(i) + ".png", phi_save)

        # phi_added = ((-1.0 * phi_coef) * (phi <= 0)) + (phi_coef * (phi > 0))



        # plt.subplot(121)
        # plt.title("phi")
        # plt.imshow(phi)
        # plt.subplot(122)
        # plt.title("phi update")
        # plt.imshow(phi_added)

        # if (i % 20 == 0 and i != scene):
        #     mean_pipe, std_pipe, mean_bg, std_bg, prob_p, prob_bg = distribution_params.cal_params(img_denoise, phi_save, i)
        #     print(mean_pipe, std_pipe, mean_bg, std_bg)

        # poly_function_pre = poly_function
        # x_pre = poly_params[1]

if __name__ == "__main__":
    bool_suc = process()