import cv2
import numpy as np
import tensorflow as tf
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.stats import gamma

def binary_activation(phi_tf, epsilon, rows, cols):
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

def energy(img, phi, img_g_tf, gm_p, gm_bg, epsilon, eta, supp):
    rows, cols = img.shape
    phi_tf = phi
    # phi_tf = tf.multiply(phi, phi_)
    a = tf.reshape(phi_tf, shape=[-1, rows, cols, 1])
    """laplacian"""
    filter_laplacian = tf.reshape(tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], tf.float32), shape=[3, 3, 1, 1])
    lap = tf.nn.conv2d(a, filter_laplacian, strides=[1, 1, 1, 1], padding='SAME')
    lap = tf.reshape(lap, shape=[rows, cols])
    """gradient sobel"""
    x_weight = tf.reshape(tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], tf.float32), shape=[3, 3, 1, 1])
    y_weight = tf.reshape(tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], tf.float32), shape=[3, 3, 1, 1])
    # x_weight = tf.reshape(tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32), shape=[3, 3, 1, 1])
    # y_weight = tf.reshape(tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], tf.float32), shape=[3, 3, 1, 1])
    grad_x = tf.nn.conv2d(a, x_weight, strides=[1, 1, 1, 1], padding='SAME')
    grad_y = tf.nn.conv2d(a, y_weight, strides=[1, 1, 1, 1], padding='SAME')
    grad_x_re = tf.reshape(grad_x, shape=[rows, cols])
    grad_y_re = tf.reshape(grad_y, shape=[rows, cols])
    """magnitude gradient"""
    mag_tf = (tf.pow(grad_x_re, 2)+tf.pow(grad_y_re, 2))
    thres = tf.constant(np.zeros([rows, cols]), tf.float32)
    cond = tf.equal(mag_tf, thres)
    mag = tf.where(cond, tf.ones([rows, cols]), mag_tf)
    # """Try"""
    # diff_rate = 1 - (1 / mag)
    # grad_flow_x = tf.mul
    # print(grad_y)
    """Check here!!!!!!!!!!!!!!!!!!!!"""
    mag = tf.reshape(mag, shape=[-1, rows, cols, 1])
    """grad / magnitude"""
    norm_x = tf.divide(grad_x, mag)
    norm_y = tf.divide(grad_y, mag)
    norm_x_re = tf.reshape(norm_x, shape=[rows, cols])
    norm_y_re = tf.reshape(norm_y, shape=[rows, cols])

    fx = tf.nn.conv2d(norm_x, x_weight, strides=[1, 1, 1, 1], padding='SAME')
    fy = tf.nn.conv2d(norm_y, y_weight, strides=[1, 1, 1, 1], padding='SAME')
    fx = tf.reshape(fx, shape=[rows, cols])
    fy = tf.reshape(fy, shape=[rows, cols])
    div1 = tf.add(fx, fy)
    inner = tf.subtract(lap, div1)


    delta = binary_activation(phi_tf, epsilon, rows, cols)
    norm_x_re = tf.reshape(norm_x, shape=[rows, cols])
    norm_y_re = tf.reshape(norm_y, shape=[rows, cols])
    norm_x_g = tf.multiply(img_g_tf, norm_x_re)
    norm_y_g = tf.multiply(img_g_tf, norm_y_re)
    norm_x_g = tf.reshape(norm_x_g, shape=[-1, rows, cols, 1])
    norm_y_g = tf.reshape(norm_y_g, shape=[-1, rows, cols, 1])
    fx = tf.nn.conv2d(norm_x_g, x_weight, strides=[1, 1, 1, 1], padding='SAME')
    fy = tf.nn.conv2d(norm_y_g, y_weight, strides=[1, 1, 1, 1], padding='SAME')
    fx = tf.reshape(fx, shape=[rows, cols])
    fy = tf.reshape(fy, shape=[rows, cols])
    div2 = tf.add(fx, fy)
    exter_length = tf.multiply(delta, div2)

    exter_area = tf.multiply(img_g_tf, delta)

    img_1D = tf.reshape(img, [-1])
    thres_img = tf.constant(11 * np.ones(rows * cols), tf.float32)
    cond = tf.greater(img_1D, thres_img)
    img_1D = tf.where(cond, img_1D, thres_img)
    img_1D = tf.cast(img_1D, tf.int32)
    print(img_1D)
    # print(gm_p)
    eta_arr = tf.constant((eta) * np.ones(rows * cols), tf.float32)
    pipe_mapped = tf.gather(gm_p, img_1D)
    bg_mapped = tf.gather(gm_bg, img_1D)
    bg_mapped_weighted = tf.add(bg_mapped, eta_arr)
    gamma_1D = tf.subtract(pipe_mapped, bg_mapped_weighted)

    # weigthed_thres_1D = tf.add(gamma_1D, eta_arr)  #"""FOR ADD WEIGTH
    # gamma = tf.reshape(weigthed_thres_1D, shape=[rows, cols])  #"""FOR ADD WEIGTH
    gamma = tf.reshape(gamma_1D, shape=[rows, cols])

    observe = tf.multiply(delta, gamma)
    # observe = gamma

    observe2 = tf.multiply(delta, supp)

    debug = gamma


    return inner, exter_length, exter_area, observe, observe2, debug



"""Try!!"""
def suppress(img_gray):  ##change to int
    # img_gray = img_gray.astype(dtype=np.uint8)
    all_size = 45  # 45
    win_size = 27  # 25
    guard_size = int((all_size - win_size) / 2)

    half_win = int(win_size / 2)

    # cv2.imshow("original", img_denoise_sh)

    mask = np.zeros([win_size, win_size], np.float32)
    guard_coeff = 1. / (pow(all_size, 2) - pow(win_size, 2))
    guard_kernel = guard_coeff * cv2.copyMakeBorder(mask, guard_size, guard_size, guard_size, guard_size,
                                                    cv2.BORDER_CONSTANT, value=1)
    guard_kernel[all_size - guard_size:all_size, 0: all_size] = 0
    thres = 1.8  # 1.2: norm window, 1.8: new window

    win_coeff = 1. / pow(win_size, 2)
    window_kernel = win_coeff * np.ones([win_size, win_size], np.float32)

    win_mean = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=window_kernel)
    guard_mean = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=guard_kernel)

    ratio = win_mean / guard_mean

    out = (ratio * (ratio >= thres)) + ((-1 * ratio) * (ratio < thres))

    return out


def levelset_cal(img_gray, phi_first, iterations, gamma_param, index, eta):

    # plt.title("image")
    # plt.imshow(img_gray)
    # plt.show()

    mew = 0.05  # 0.05
    lamda = 50.0 # 50.0
    v = -5.0 #-5.0 #initial inside: -
    beta = 0.1 #0.1
    alpha = 0.5 #0.5, 0.3
    epsilon = 1.0 # 1.0
    time = 4.0 # 4.0

    """reduce alpha term"""
    if(index!=10):
        beta = 0.0

    coeff1 = time * mew
    coeff2 = time * lamda
    coeff3 = time * v
    coeff4 = time * beta
    coeff5 = time * alpha
    # coeff5 = time * alpha
    # print("coeff1: %.2f, coeff2: %.2f, coeff3: %.2f, coeff4: %.2f" % (coeff1, coeff2, coeff3, coeff4))


    """use edge from CAFAR instead g from original image"""
    x_axis = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    y_axis = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
    edge = np.sqrt(np.square(x_axis) + np.square(y_axis))
    g = 1.0 / (1.0 + (edge ** 2))

    """gamma distribution"""
    x = np.arange(0, 256, 1)
    g1 = gamma.logpdf(x=x, a=gamma_param[0][0], loc=gamma_param[0][1], scale=gamma_param[0][2])
    g2 = gamma.logpdf(x=x, a=gamma_param[1][0], loc=gamma_param[1][1], scale=gamma_param[1][2])

    """Try!!"""
    out = suppress(img_gray)

    phi_init = tf.Variable(phi_first, dtype=tf.float32)
    img_tf = tf.Variable(img_gray,  dtype=tf.float32)
    img_g_tf = tf.Variable(g, dtype=tf.float32)
    gamma_pipe = tf.Variable(g1, dtype=tf.float32)
    gamma_bg = tf.Variable(g2, dtype=tf.float32)
    supp = tf.Variable(out, dtype=tf.float32)

    L_phi = energy(img_tf, phi_init, img_g_tf, gamma_pipe, gamma_bg, epsilon, eta, supp)


    phi_old = phi_first

    # plt.title("initial phi")
    # plt.imshow(phi_old)
    # plt.show()
    plt.ion()
    with tf.Session() as sess:
        # Initiate session and initialize all vaiables
        sess.run(tf.global_variables_initializer())
        for i in range(iterations+1):
            print("times#: ", i)
            # img = sess.run(img_tf)
            print("phi_max: %f, phi_min: %f" % (phi_old.max(), phi_old.min()))
            # print("img: ", img.max())


            penalty, length, area, observe, observe2, debug = sess.run(L_phi,feed_dict={img_tf: img_gray, phi_init: phi_old, img_g_tf: g,
                                                                        gamma_pipe: g1, gamma_bg: g2, supp:out})
            # plt.title("delta")
            # plt.imshow(debug)
            # plt.show()


            """Try!!"""
            all_energy = (coeff1 * penalty) + (coeff2 * length) + (coeff3 * area) + (coeff4 * observe) + (coeff5 * observe2)
            phi_new = phi_old + (all_energy)
            phi_new = phi_new.astype('float32')
            phi_old = phi_new
            # print(phi_new)
            # plt.title("penalty")
            # plt.imshow(penalty)
            # plt.show()
            print("a: %.5f, b: %.5f, c: %.5f, d: %.5f, e: %.5f" % ((coeff1 * penalty).max(), (coeff2 * length).max(), (coeff3 * area).max(),(coeff4 * observe).max(), (coeff5 * observe2).max()))
            # """show result after iteration"""
            # if (i % iterations == 0) and (i != 0):
            #     plt.title("result")
            #     plt.imshow(img_gray, cmap='gray')
            #     plt.title("frame"+str(index))
            #     # plt.imshow(bayes)
            #     CS = plt.contour(phi_old, 0, colors='r', linewidths=2)
            #     plt.draw()
            #     # plt.savefig(r"C:\Users\mook\Desktop\result_feature_phi_reinit_RTheta_img_" + str(index))
            #     plt.show()
            #     plt.pause(0.05)
            #
            #     # plt.title("gamma")
            #     # plt.imshow(debug)
            #     # plt.show()
            #
            #     plt.clf()
            #     plt.close()
            print()
            plt.ioff()
    return phi_old