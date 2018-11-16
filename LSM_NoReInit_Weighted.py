import cv2
import numpy as np
import denoise_normal
import levelset_weighted
import matplotlib.pyplot as plt

def process ():
    rootpath = r"C:\Users\mook\PycharmProjects\LSM\images\RTheta_img_"

    """DENOISE Parameters"""
    start_range = 0.0
    stop_range = 30.0

    """CAFAR Parameters"""
    box_size = 41
    guard_size = 31
    pfa = 0.2
    size = 3
    morp_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))

    """gamma params"""
    shape_pipe = 33811.17
    loc_pipe = -6640.13
    scale_pipe = 0.20

    shape_bg = 58.79
    loc_bg = -125.25
    scale_bg = 3.31

    gamma_param = [[shape_pipe, loc_pipe, scale_pipe], [shape_bg, loc_bg, scale_bg]]

    """level set"""
    bool_count = True
    iteration = 1000

    """mask-manual for eta term"""
    # mask_bg = cv2.imread(r"C:\Users\RSDFL-PC01\PycharmProjects\Mook work\mask_bg.png", 0)
    # mask_bg = (1 * (mask_bg==255)) + (0 * (mask_bg==0))
    # print("mask: ", mask_bg.max(), mask_bg.min())
    prob_p = 13797./364000
    # prob_p = 20000./3864000
    prob_bg = 350200./364000
    # prob_bg = 360000./364000
    eta = 1.0 * np.log10(prob_bg / prob_p)
    print("eta: ", eta)
    # eta_mat = (np.log10(prob_bg / prob_p)) * mask_bg
    # print(eta_mat.min())
    # eta = np.log10(prob_p / prob_bg)

    for i in range (100, 130): #10, 130
        imagepath = rootpath + str(i) + ".jpg"
        # imagepath = rootpath
        print (imagepath)
        """_____IMPORT IMAGE_____"""
        img = cv2.imread(imagepath)
        img = img[25:525, 20:748] # Crop
        img_rgb = img
        img = cv2.GaussianBlur(img,(9,9),3)
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
        # cv2.imshow("img", img)
        """______DENOISE______"""
        img_denoise = denoise_normal.denoise_range(img, start_range, stop_range)
        img_denoise_sh = img_denoise.astype(dtype = np.uint8)
        # cv2.imshow("img_denoise", img_denoise_sh)

        """Level-set"""
        if (bool_count):
            bool_count = False
            phi_coef = 6.0 #6.0
            init_phi = (-1.0 * phi_coef) * np.ones(img_denoise_sh.shape, 'float32')
            # init_phi[200:450, 450:700] = phi_coef
            init_phi[200:450, 450:700] = phi_coef
            print ("FIRST FRAME")
        else:
            init_phi = phi
            print("FRAME ", i)
            iteration = 300
            """suppress"""
            # eta_mat = 0 * eta_mat


        phi = levelset_weighted.levelset_cal(img_denoise, init_phi, iteration, gamma_param, i, eta)
        # cv2.imshow("img_pipe", phi)
        phi_thres = (255 * (phi>0.0)) + (0 * (phi<=0.0))
        cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\phi_" + str(i) + "NewWin_ReduceBeta.png", phi_thres)
        # # print(phi_thres.min())
        # plt.title("phi")
        # plt.imshow(phi)
        # plt.show()
        phi_thres = phi_thres.astype(np.uint8)
        # size = 3
        # morp_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        """initial for feature drawing (phi_lim)"""
        phi_lim = 255 * np.ones(img_denoise_sh.shape, 'float32')
        _, thresh = cv2.threshold(phi_thres, 127, 255, cv2.THRESH_BINARY)
        # img_morp = cv2.dilate(thresh, morp_kernel, iterations=2)
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = contours
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        """Allow just 5 parts of pipe"""
        for j in range (0,len(cnts)):
            if (len(cnts[j]) >= 5):
                ellipse = cv2.fitEllipse(cnts[j])
                img = cv2.ellipse(img, ellipse, (0, 255, 0), 2)
                """cond to filted a contour in phi_lim"""
                print ("MA, ma: ", ellipse[1])
                # M = cv2.moments(cnts[j])
                # cX = int(M["m10"] / M["m00"])
                # cY = int(M["m01"] / M["m00"])
                area = cv2.contourArea(cnts[j])
                if (ellipse[1][1] >= 100 and area >= 150):
                # if (ellipse[1][0] >= 50 or ellipse[1][1] >= 50):
                    cv2.drawContours(phi_lim, [cnts[j]], -1, (0, 255, 0), -1)
                    cv2.drawContours(img_rgb, [cnts[j]], -1, (0, 0, 255), 3)
                    # cv2.putText(img_rgb, str(ellipse[1][1]), (cX, cY), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (170, 128, 255), 2)
                    # print("area: ", cv2.contourArea(cnts[j]))
                    # cv2.putText(img_rgb, cv2.contourArea(cnts[j]), (cX, cY), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (170, 128, 255), 2)
        # cv2.imshow("contour", img_rgb)
        # cv2.waitKey(-1)

        _, phi_lim = cv2.threshold(phi_lim, 127, 255, cv2.THRESH_BINARY_INV)
        phi_lim = (1 * (phi_lim == 255)) + (0 * (phi_lim == 0))
        phi_masked = phi * phi_lim
        cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\result_" + str(i) + "NewWin_ReduceBeta.png", img_rgb)
        # # plt.title("phi_masked")
        # # plt.imshow(phi_masked)
        # # plt.show()
        phi = (-6 * (phi_masked == 0)) + (phi_masked * (phi_masked != 0))


if __name__ == "__main__":
    bool_suc = process()