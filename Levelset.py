import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from time import sleep
from scipy.special import erf
from ParamsSetting import levelsetparams
from InitialPhiWithCACFAR import phiInitialization

def cacfarRatio(all_size, win_size, img_gray):
    guard_size = int((all_size - win_size) / 2)
    mask = np.zeros([win_size, win_size], np.float32)
    ref_coeff = 1. / (pow(all_size, 2) - pow(win_size, 2))
    ref_kernel = ref_coeff * cv2.copyMakeBorder(mask, guard_size, guard_size, guard_size, guard_size,
                                                    cv2.BORDER_CONSTANT, value=1)
    ref_kernel[guard_size:all_size-guard_size, guard_size: all_size-guard_size] = 0
    testing_kernel = np.zeros(ref_kernel.shape, np.float32)
    """testing cell size 1"""
    testing_kernel[int((all_size + 1) / 2), int((all_size + 1) / 2)] = 1
    testing_cell = 1
    testing_pixel = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=testing_kernel)
    ref_mean = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=ref_kernel)
    ratio = testing_pixel / ref_mean
    return ratio, testing_cell

def binaryActivation(phi_tf, epsilon, rows, cols):
    ab_phi = tf.abs(phi_tf)
    eps = tf.constant(epsilon * np.ones([rows, cols]), tf.float32)
    pi_eps = tf.constant((np.pi/epsilon)*np.ones([rows, cols]), tf.float32)

    param1 = tf.constant((1.0/(2.0*epsilon)) * np.ones([rows, cols]), tf.float32)
    coef = tf.multiply(pi_eps, phi_tf)
    param_1 = tf.cos(coef)
    param_2 = tf.constant(np.ones([rows, cols]), tf.float32) + param_1
    param_3 = tf.multiply(param1, param_2)
    cond = tf.greater(ab_phi, eps)
    out = tf.where(cond, tf.zeros([rows, cols]), param_3)
    return out

def energyComputation(img, phi_tf, g_tf, epsilon, obs):
    rows, cols = img.shape
    filter_laplacian = tf.reshape(tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], tf.float32), shape=[3, 3, 1, 1])
    """padding kernel [#r_left, #r_right], [#c_left, #c_right]"""
    paddings = tf.constant([[1, 1], [1, 1]])
    """pad phi"""
    phi_pad = tf.pad(phi_tf, paddings, "SYMMETRIC")  # [rows, cols]
    rows_pad, cols_pad = phi_pad.shape
    """reshape from [rows, cols] to be a shape [batch, h, w, channel] input of convolution"""
    phi_pad = tf.reshape(phi_pad, shape=[-1, rows_pad, cols_pad, 1])
    """convolution: laplacian"""
    lap = tf.nn.conv2d(phi_pad, filter_laplacian, strides=[1, 1, 1, 1], padding='VALID')
    lap = tf.reshape(lap, shape=[rows, cols])

    """kernel of gradient"""
    y_weight = tf.reshape(tf.constant([-0.5, 0, +0.5], tf.float32), [3, 1, 1, 1])
    x_weight = tf.reshape(y_weight, [1, 3, 1, 1])
    """padding kernel [#r_left, #r_right], [#c_left, #c_right]"""
    paddings_y = tf.constant([[1, 1], [0, 0]])
    paddings_x = tf.constant([[0, 0], [1, 1]])
    """pad phi"""
    phi_y_pad = tf.pad(phi_tf, paddings_y, "SYMMETRIC")  # [rows, cols]
    phi_x_pad = tf.pad(phi_tf, paddings_x, "SYMMETRIC")  # [rows, cols]
    rows_pad_y, cols_pad_y = phi_y_pad.shape
    rows_pad_x, cols_pad_x = phi_x_pad.shape
    """reshape from [rows, cols] to be a shape [batch, h, w, channel] input of convolution"""
    phi_y_pad = tf.reshape(phi_y_pad, shape=[-1, rows_pad_y, cols_pad_y, 1])
    phi_x_pad = tf.reshape(phi_x_pad, shape=[-1, rows_pad_x, cols_pad_x, 1])
    """convolution: gradient x, and y [batch, h, w, channel]"""
    grad_y = tf.nn.conv2d(phi_y_pad, y_weight, [1, 1, 1, 1], 'VALID')
    grad_x = tf.nn.conv2d(phi_x_pad, x_weight, [1, 1, 1, 1], 'VALID')
    """reshape from [batch, h, w, channel] to be a shape [rows, cols] for debug(display)"""
    grad_y_re = tf.reshape(grad_y, shape=[rows, cols])
    grad_x_re = tf.reshape(grad_x, shape=[rows, cols])
    """magnitude gradient [rows, cols]"""
    mag_tf = tf.sqrt(tf.pow(grad_x_re, 2) + tf.pow(grad_y_re, 2))
    thres = tf.constant(np.zeros([rows, cols]), tf.float32)
    cond = tf.equal(mag_tf, thres)
    mag = tf.where(cond, tf.ones([rows, cols]), mag_tf)
    """reshape from [rows, cols] to be a shape [batch, h, w, channel]"""
    mag_re = tf.reshape(mag, shape=[-1, rows, cols, 1])
    """grad / magnitude [batch, h, w, channel]"""
    norm_y = tf.divide(grad_y, mag_re)
    norm_x = tf.divide(grad_x, mag_re)
    """reshape from [batch, h, w, channel] to be a shape [rows, cols]"""
    norm_y_re = tf.reshape(norm_y, shape=[rows, cols])
    norm_x_re = tf.reshape(norm_x, shape=[rows, cols])
    """pad norm x, and y"""
    norm_y_pad = tf.pad(norm_y_re, paddings_y, "SYMMETRIC")  # [rows, cols]
    norm_x_pad = tf.pad(norm_x_re, paddings_x, "SYMMETRIC")  # [rows, cols]
    """reshape from [rows, cols] to be a shape [batch, h, w, channel] input of convolution"""
    norm_y_pad = tf.reshape(norm_y_pad, shape=[-1, rows_pad_y, cols_pad_y, 1])
    norm_x_pad = tf.reshape(norm_x_pad, shape=[-1, rows_pad_x, cols_pad_x, 1])
    """convolution: gradient norm x, and norm y [batch, h, w, channel]"""
    fy = tf.nn.conv2d(norm_y_pad, y_weight, [1, 1, 1, 1], 'VALID')
    fx = tf.nn.conv2d(norm_x_pad, x_weight, [1, 1, 1, 1], 'VALID')
    # print("fx: ", fx.shape)
    fxx = tf.reshape(fx, shape=[rows, cols])
    fyy = tf.reshape(fy, shape=[rows, cols])
    div1 = tf.add(fxx, fyy)
    inner = tf.subtract(lap, div1)

    delta = binaryActivation(phi_tf, epsilon, rows, cols)
    norm_x_re = tf.reshape(norm_x, shape=[rows, cols])
    norm_y_re = tf.reshape(norm_y, shape=[rows, cols])
    norm_x_g = tf.multiply(g_tf, norm_x_re)
    norm_y_g = tf.multiply(g_tf, norm_y_re)
    norm_x_g_re = tf.reshape(norm_x_g, shape=[-1, rows, cols, 1])
    norm_y_g_re = tf.reshape(norm_y_g, shape=[-1, rows, cols, 1])
    fx = tf.image.image_gradients(norm_x_g_re)
    fy = tf.image.image_gradients(norm_y_g_re)
    fyx, fxx = tf.split(fx, num_or_size_splits=2, axis=0)
    fyy, fxy = tf.split(fy, num_or_size_splits=2, axis=0)
    fxx = tf.reshape(fxx, shape=[rows, cols])
    fyy = tf.reshape(fyy, shape=[rows, cols])
    div2 = tf.add(fxx, fyy)
    exter_length = tf.multiply(delta, div2)

    exter_area = tf.multiply(g_tf, delta)

    norm_x_obs = tf.multiply(obs, norm_x_re)
    norm_y_obs = tf.multiply(obs, norm_y_re)
    norm_x_obs_re = tf.reshape(norm_x_obs, shape=[-1, rows, cols, 1])
    norm_y_obs_re = tf.reshape(norm_y_obs, shape=[-1, rows, cols, 1])
    fx_obs = tf.image.image_gradients(norm_x_obs_re)
    fy_obs = tf.image.image_gradients(norm_y_obs_re)
    fyx_obs, fxx_obs = tf.split(fx_obs, num_or_size_splits=2, axis=0)
    fyy_obs, fxy_obs = tf.split(fy_obs, num_or_size_splits=2, axis=0)
    fxx_obs = tf.reshape(fxx_obs, shape=[rows, cols])
    fyy_obs = tf.reshape(fyy_obs, shape=[rows, cols])

    div3 = tf.add(fxx_obs, fyy_obs)
    observe2 = tf.multiply(tf.multiply(mag, div3), delta)

    return inner, exter_length, exter_area, observe2

def updatePhi(img_gray, levelset_param, num):
    phiInit = levelset_param[0]
    iterations = levelset_param[1]
    mew = levelset_param[2]
    lamda = levelset_param[3]
    v = levelset_param[4]
    alpha = levelset_param[5]
    epsilon = levelset_param[6]
    time = levelset_param[7]

    coeff1 = time * mew
    coeff2 = time * lamda
    coeff3 = time * v
    coeff5 = time * alpha

    CACFAR, testing_cell = cacfarRatio(levelset_param[8], levelset_param[9], img_gray)
    CACFAR = (0.1 * (CACFAR < 0.1)) + (CACFAR * (CACFAR >= 0.1))
    g = 1 / CACFAR
    obse = g

    phi_init = tf.placeholder(tf.float32, shape=phiInit.shape)
    img_tf = tf.constant(img_gray, dtype=tf.float32, shape=img_gray.shape)
    g_tf = tf.constant(g, dtype=tf.float32, shape=g.shape)
    obs = tf.constant(obse, dtype=tf.float32, shape=obse.shape)
    L_phi = energyComputation(img_tf, phi_init, g_tf, epsilon, obs)
    phiPrevious = phiInit
    with tf.Session() as sess:
        print("FRAME ", num)
        print("times#: ", end='')
        # plt.ion()
        for i in range(iterations + 1):
            j = (i + 1) / iterations
            sys.stdout.write('\r')
            penalty, length, area, observe2 = sess.run(L_phi, feed_dict={phi_init: phiPrevious})
            all_energy = (coeff1 * penalty) + (coeff2 * length) + (coeff3 * area) + (coeff5 * observe2)
            phiUpdated = phiPrevious + (all_energy)
            phiUpdated = phiUpdated.astype('float32')
            phiPrevious = phiUpdated
            sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
            sys.stdout.flush()
            sleep(0.05)
        #     plt.title("levelset" + str(num))
        #     plt.imshow(img_gray, cmap='gray')
        #     CS = plt.contour(phiPrevious, 0, colors='r', linewidths=2)
        #     plt.draw()
        #     plt.show()
        #     plt.pause(0.05)
        #     plt.clf()
        # plt.ioff()
        # plt.close()
    return phiPrevious


def levelsetProcess(img, phiInit, ReInitPhi, shapePrev, num):
    phiCoef = levelsetparams.phi_coef
    iterCurved = levelsetparams.iteration_curved
    iterStraight = levelsetparams.iteration_straight
    mewCoef = levelsetparams.penalty_mew
    lambdaCoef = levelsetparams.length_lambda
    vCoef = levelsetparams.area_v
    alphaCoef = levelsetparams.GAC_alpha
    epsilon = levelsetparams.epsilon
    tau = levelsetparams.step_tau

    refWin = levelsetparams.referenceCell_win
    guardWin = levelsetparams.guardingCell_win
    pfa = levelsetparams.pfa
    angleCurved = levelsetparams.angle_limit_curved
    angleStraight = levelsetparams.angle_limit_straight
    majorLenght = levelsetparams.major_axis_length
    print("\nLevelset process of frame %d..." %num)
    if(phiInit is None or ReInitPhi):
        im = img.astype(dtype=np.uint8)
        print("Initial phi with CACFAR method")
        shape, cacfar, res = phiInitialization(im, refWin, guardWin, pfa, angleCurved,
                                               angleStraight, majorLenght)
        phiInit = res[0]
        ellipse = res[1]
        phiInit = phiCoef * phiInit  # -phiCoef < phi < phiCoef
    else:
        shape = shapePrev
        print ("Use the iteration from the previous frame")

    if (shape == "right" or shape == "left"):
        iteration = iterCurved
    elif (shape == "straight"):
        iteration = iterStraight
    else:
        print("The shape of a pipeline is incorrect.")
        sys.exit()

    # plt.imshow(phiInit, 'gray')
    # plt.show()
    levelset_param = [phiInit, iteration, mewCoef, lambdaCoef, vCoef, alphaCoef, epsilon, tau, refWin, guardWin]
    phi = updatePhi(img, levelset_param, num)
    return phi, shape



