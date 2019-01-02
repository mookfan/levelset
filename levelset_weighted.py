import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
from scipy.special import erf
# import levelset_normal

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

def energy(img, phi, img_g_tf, gm_p, gm_bg, epsilon, eta, obs):
    rows, cols = img.shape
    phi_tf = phi
    # a = tf.reshape(phi_tf, shape=[-1, rows, cols, 1])
    # """laplacian"""
    filter_laplacian = tf.reshape(tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], tf.float32), shape=[3, 3, 1, 1])
    # lap = tf.nn.conv2d(a, filter_laplacian, strides=[1, 1, 1, 1], padding='SAME')
    # lap = tf.reshape(lap, shape=[rows, cols])

    """padding kernel [#r_left, #r_right], [#c_left, #c_right]"""
    paddings = tf.constant([[1, 1], [1, 1]])
    """pad phi"""
    phi_pad = tf.pad(phi, paddings, "SYMMETRIC") #[rows, cols]
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
    phi_y_pad = tf.pad(phi, paddings_y, "SYMMETRIC") #[rows, cols]
    phi_x_pad = tf.pad(phi, paddings_x, "SYMMETRIC") #[rows, cols]
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

    # """use tf.image to find gradient"""
    # phi_re = tf.reshape(phi_tf, shape=[-1, rows, cols, 1])
    # grad = tf.image.image_gradients(phi_re)
    # print(grad)
    # grad_y, grad_x = tf.split(grad, num_or_size_splits=2, axis=0)
    # grad_x = tf.squeeze(grad_x, [0])
    # grad_y = tf.squeeze(grad_y, [0])

    """magnitude gradient [rows, cols]"""
    mag_tf = tf.sqrt(tf.pow(grad_x_re, 2)+tf.pow(grad_y_re, 2))
    thres = tf.constant(np.zeros([rows, cols]), tf.float32)
    cond = tf.equal(mag_tf, thres)
    mag = tf.where(cond, tf.ones([rows, cols]), mag_tf)
    """reshape from [rows, cols] to be a shape [batch, h, w, channel]"""
    mag = tf.reshape(mag, shape=[-1, rows, cols, 1])
    """grad / magnitude [batch, h, w, channel]"""
    norm_y = tf.divide(grad_y, mag)
    norm_x = tf.divide(grad_x, mag)
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

    delta = binary_activation(phi_tf, epsilon, rows, cols)
    norm_x_re = tf.reshape(norm_x, shape=[rows, cols])
    norm_y_re = tf.reshape(norm_y, shape=[rows, cols])
    norm_x_g = tf.multiply(img_g_tf, norm_x_re)
    norm_y_g = tf.multiply(img_g_tf, norm_y_re)
    norm_x_g = tf.reshape(norm_x_g, shape=[-1, rows, cols, 1])
    norm_y_g = tf.reshape(norm_y_g, shape=[-1, rows, cols, 1])
    fx = tf.image.image_gradients(norm_x_g)
    fy = tf.image.image_gradients(norm_y_g)
    fyx, fxx = tf.split(fx, num_or_size_splits=2, axis=0)
    fyy, fxy = tf.split(fy, num_or_size_splits=2, axis=0)
    # fx = tf.nn.conv2d(norm_x_g, x_weight, strides=[1, 1, 1, 1], padding='SAME')
    # fy = tf.nn.conv2d(norm_y_g, y_weight, strides=[1, 1, 1, 1], padding='SAME')
    fxx = tf.reshape(fxx, shape=[rows, cols])
    fyy = tf.reshape(fyy, shape=[rows, cols])
    div2 = tf.add(fxx, fyy)
    exter_length = tf.multiply(delta, div2)

    exter_area = tf.multiply(img_g_tf, delta)

    observe2 = tf.multiply(delta, obs)

    debug = grad_y_re

    return inner, exter_length, exter_area, observe2, debug

def ratio_cal(gauss_param, win_size, all_size, testing_cell):

    mean_p, stand_p = gauss_param[0][0], gauss_param[0][1]
    mean_bg, stand_bg = gauss_param[1][0], gauss_param[1][1]
    scale = 0.01
    hor_axis =np.arange(0.0, 10.0, scale)
    """calculate ratio gaussian p(z) by given the relationship z = x/y
        x: gamma distribution of pipe
        y: gamma distribution of background
        z = ratio between pipe/background"""
    for i in range (0, 2):
        """object/background"""
        if (i==0):
            std_p = stand_p/np.sqrt(testing_cell*testing_cell)
            std_bg = stand_bg/np.sqrt((all_size*all_size)-(win_size*win_size))

        else:
            """background/background"""
            mean_p = mean_bg
            std_p = stand_bg/np.sqrt(testing_cell*testing_cell)
            std_bg = stand_bg/np.sqrt((all_size*all_size)-(win_size*win_size))

        a = (np.power(hor_axis, 2)*np.power(std_bg, 2))+(np.power(std_p, 2))
        b = (2*hor_axis*mean_p*np.power(std_bg, 2))+(2*mean_bg*np.power(std_p, 2))
        c = (np.power(mean_p, 2)*np.power(std_bg, 2))+(np.power(mean_bg, 2)*np.power(std_p, 2))

        a = a.astype(float)
        b = b.astype(float)
        c = c.astype(float)

        k1 = (-1./(2*np.power(std_p, 2)*np.power(std_bg, 2)))
        k2 = (-1*np.power(b/(2*a), 2)*a)
        k = np.exp(k1*(k2+c))
        k = k.astype(float)
        coeff = (k/(2*np.pi*std_p*std_p))

        add = (b/(2*a))
        pow_exp = (a/(2*np.power(std_p, 2)*np.power(std_bg, 2)))
        """proof integrate part"""
        # res1 = quad(integrand, np.NINF, np.PINF, args=(add,pow_exp))
        # result1 = coeff*res1[0]
        # print("integral's result: ", result1)


        res2 = ((1./(pow_exp))*np.exp(-(pow_exp*np.power(add, 2))))+((add*np.sqrt((np.pi)/pow_exp))*erf(np.sqrt(pow_exp)*add))
        result2 = coeff*res2
        if(i==0):
            res_obj_bg=result2
            # plt.subplot(121)
            # plt.title("p(z) obj_bg")
            # plt.plot(hor_axis, result2, linewidth=1)
            # loc_thres = np.where(result2 == result2.max())
        else:
            res_bg_bg=result2
            # plt.subplot(122)
            # plt.title("p(z) bg_bg")
            # plt.plot(hor_axis, result2, linewidth=1)
            # loc_thres = np.where(result2 == result2.max())
    # print(loc_thres[0][0]*scale)
    # print(result2[loc_thres[0]])
    # plt.show()
    return res_obj_bg, res_bg_bg

def suppress(img_gray, gauss_param):  ##change to int
    all_size = 51 # 41
    win_size = 41  # 31
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
    # """testing cell size 2n+1"""
    # n=4
    # testing_cell = (2*n)+1
    # testing_kernel[int((all_size + 1) / 2)-n:int((all_size + 1) / 2)+n, int((all_size + 1) / 2)-n:int((all_size + 1) / 2)+n] = 1./(pow((2*n)+1, 2))

    # plt.subplot(121)
    # plt.title("reference cell")
    # plt.imshow(ref_kernel)
    # plt.subplot(122)
    # plt.title("testing cell")
    # plt.imshow(testing_kernel)
    # plt.show()


    testing_pixel = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=testing_kernel)
    ref_mean = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=ref_kernel)

    ratio = testing_pixel / ref_mean
    print("debug: ", ratio.max(), ratio.min())
    # plt.title("ratio")
    # plt.imshow(ratio)
    # plt.show()
    # print(ratio.max(), ratio.min())
    # plt.subplot(221)
    # plt.title("testing")
    # plt.imshow(testing_pixel, 'gray')
    # plt.subplot(222)
    # plt.title("reference")
    # plt.imshow(ref_mean, 'gray')
    # plt.subplot(223)
    # plt.title("ratio")
    # plt.imshow(ratio)
    # plt.show()

    res_obj_bg, res_bg_bg = ratio_cal(gauss_param, win_size, all_size, testing_cell)

    loc_thres = np.where(res_obj_bg == res_obj_bg.max())
    # print("location of (obj/bg) max: ", loc_thres[0])

    # ratio = ratio.astype(np.uint8)
    # plt.title("ratio")
    # plt.imshow(ratio)
    # plt.show()
    """use uint16 because some of them has their value more than 255"""
    out_obj_bg = (res_obj_bg[(ratio * 100).astype(np.uint16)] * (ratio >= 0))
    # out_obj_bg = (res_obj_bg[(ratio*100).astype(np.uint16)]*((ratio*100).astype(np.uint16)<loc_thres[0])) + (res_obj_bg.max()*((ratio*100).astype(np.uint16)>=loc_thres[0]))

    # plt.subplot(121)
    # test=(ratio * 100).astype(np.uint16)
    # plt.title("testttt")
    # plt.imshow(test, 'gray')
    # plt.subplot(122)
    # plt.title("ratio")
    # plt.imshow(ratio, 'gray')
    # plt.show()

    # hor_axis = np.arange(0.0, 5.0, 0.01)
    # plt.subplot(221)
    # plt.title("ratio")
    # plt.imshow(ratio, 'gray')
    # plt.subplot(222)
    # plt.title("map ratio obj/background")
    # plt.imshow(out_obj_bg, 'gray')
    # plt.subplot(223)
    # plt.title("map obj/background")
    # plt.plot(hor_axis, res_obj_bg, 'gray')
    # plt.show()

    loc_thres = np.where(res_bg_bg == res_bg_bg.max())
    # print("location of (bg/bg) max: ", loc_thres[0])
    out_bg_bg = (res_bg_bg[(ratio * 100).astype(np.uint16)] * (ratio >= 0))
    # out_bg_bg = (res_bg_bg[(ratio*100).astype(np.uint16)]*((ratio * 100).astype(np.uint16)>loc_thres[0])) + (res_bg_bg.max()*((ratio*100).astype(np.uint16)<=loc_thres[0]))


    # plt.subplot(221)
    # plt.title("ratio")
    # plt.imshow(ratio, 'gray')
    # plt.subplot(222)
    # plt.title("map ratio background/background")
    # plt.imshow(out_bg_bg, 'gray')
    # plt.subplot(223)
    # plt.title("map bg/background")
    # plt.plot(hor_axis, res_bg_bg, 'gray')
    # plt.show()



    """to joint pipe fun"""
    # input_join = (255 * (ratio >= thres)) + (0 * (ratio < thres))
    # input_join = input_join.astype(np.uint8)
    # plt.title("threshold_CFAR")
    # plt.imshow(input_join)
    # plt.show()
    # input_join = reduce_noise(input_join)
    #
    # # input_join = cv2.cvtColor(input_join, cv2.COLOR_BGR2GRAY)
    # output_join, joint_point, rect = join_pipe.join(input_join)
    # out = (ratio * (output_join==255)) + ((-1 * ratio) * (output_join==0))
    # # plt.title("out suppress")
    # # plt.imshow(out)
    # # plt.show()
    # return out, joint_point, rect
    """return mapped ratio gaussian"""
    return out_obj_bg, out_bg_bg

def levelset_cal(img_gray, phi_first, iterations, gauss_param, index, eta, start_bool):

    # plt.title("phi")
    # plt.imshow(phi_first)
    # plt.show()

    mew = 0.1 # 0.05
    lamda = 10.0 # 50.0
    v = -5.0 #-5.0 #initial inside: -
    # beta = 0.1 #0.1
    alpha = 0.4 #0.5,  0.08 for n sized
    epsilon = 1.0 # 1.0
    time = 2.0 # 4.0

    """reduce alpha term"""
    if(start_bool==False):
        mew = 0.2
        lamda = 50.0 #-
        time = 1.0
        alpha = 0.3 #0.2

    coeff1 = time * mew
    coeff2 = time * lamda
    coeff3 = time * v
    # coeff4 = time * beta
    coeff5 = time * alpha
    # print("coeff1: %.2f, coeff2: %.2f, coeff3: %.2f, coeff4: %.2f" % (coeff1, coeff2, coeff3, coeff4))


    # plt.title("img")
    # plt.imshow(img_gray)
    # plt.show()
    """use edge denoise-image"""
    x_axis = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    y_axis = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)
    edge = np.sqrt(np.square(x_axis) + np.square(y_axis))
    g = 1.0 / (1.0 + (edge ** 2))

    # """gamma distribution"""
    x = np.arange(0, 256, 1)
    # g1 = gamma.logpdf(x=x, a=gamma_param[0][0], loc=gamma_param[0][1], scale=gamma_param[0][2])
    # g2 = gamma.logpdf(x=x, a=gamma_param[1][0], loc=gamma_param[1][1], scale=gamma_param[1][2])

    """gaussian distribution"""
    g1 = norm.pdf(x=x, loc=gauss_param[0][0], scale=gauss_param[0][1])
    g2 = norm.pdf(x=x, loc=gauss_param[1][0], scale=gauss_param[1][1])

    # out_sup, joint_point, rect = suppress(img_gray, gauss_param)
    # rect = (0*(rect==0))+(1*(rect==255))
    # print("rect max: ", rect.max())
    # if (rect.max()!=0):
    #     rect = rect
    #     out = out_sup
    # else:
    #     rect_pre = (3*(rect_pre>0))+(0*(rect_pre<0)) #boost rect_pre 3 times
    #     rect = rect_pre
    #     out = out_sup + rect
    mapped_obj_bg, mapped_bg_bg = suppress(img_gray, gauss_param)

    """avoid inf from ln(0)"""
    mapped_obj_bg = (mapped_obj_bg * (mapped_obj_bg != 0)) + (0.001 * (mapped_obj_bg == 0))
    mapped_bg_bg = (mapped_bg_bg * (mapped_bg_bg != 0)) + (0.001 * (mapped_bg_bg == 0))

    # plt.subplot(221)
    # plt.title("suppress_mapped_obj_bg")
    # plt.imshow(mapped_obj_bg)
    # plt.subplot(222)
    # plt.title("suppress_mapped_bg_bg")
    # plt.imshow(mapped_bg_bg)

    """take ln"""
    mapped_obj_bg = np.log(mapped_obj_bg)
    mapped_bg_bg = np.log(mapped_bg_bg)
    # plt.subplot(223)
    # plt.title("ln(mapped_obj_bg)")
    # plt.imshow(mapped_obj_bg)
    # plt.subplot(224)
    # plt.title("ln(mapped_bg_bg)")
    # plt.imshow(mapped_bg_bg)
    # plt.show()
    # plt.savefig(r"C:\Users\mook\PycharmProjects\LSM\experiment\ver.4\suppress_" + str(index) + ".png")
    eta_arr = eta*np.ones(mapped_bg_bg.shape)
    obse = mapped_obj_bg - (mapped_bg_bg+eta_arr)
    # out = mapped_obj_bg-(mapped_bg_bg)
    # plt.subplot(221)
    # plt.title("ln(mapped_obj_bg)")
    # plt.imshow(mapped_obj_bg)
    # plt.subplot(222)
    # plt.title("(ln(mapped_bg_bg))")
    # plt.imshow(mapped_bg_bg)
    # plt.subplot(223)
    # plt.title("ln(mapped_obj_bg)-(ln(mapped_bg_bg))")
    # plt.imshow(obse)
    # plt.show()


    # phi_init = tf.Variable(phi_first, dtype=tf.float32)
    phi_init = tf.placeholder(tf.float32, shape=phi_first.shape)
    # img_tf = tf.placeholder(tf.float32, img_gray.shape)
    # img_g_tf = tf.placeholder(tf.float32, shape=g.shape)
    # gamma_pipe = tf.placeholder(tf.float32, shape=None)
    # gamma_bg = tf.placeholder(tf.float32, shape=None)
    img_tf = tf.constant(img_gray, dtype=tf.float32, shape=img_gray.shape)
    img_g_tf = tf.constant(g, dtype=tf.float32, shape=g.shape)
    dist_pipe = tf.constant(g1, dtype=tf.float32, shape=g1.shape)
    dist_bg = tf.constant(g2, dtype=tf.float32, shape=g2.shape)
    obs = tf.constant(obse, dtype=tf.float32, shape=obse.shape)

    L_phi = energy(img_tf, phi_init, img_g_tf, dist_pipe, dist_bg, epsilon, eta, obs)


    phi_old = phi_first

    # plt.title("initial phi")
    # plt.imshow(phi_old)
    # plt.show()
    with tf.Session() as sess:
        # Initiate session and initialize all vaiables
        # sess.run(tf.global_variables_initializer())
        plt.ion()
        for i in range(iterations+1):
            print("times#: ", i)
            # img = sess.run(img_tf)
            # print("phi_max: %f, phi_min: %f" % (phi_old.max(), phi_old.min()))
            # print("img: ", img.max())


            # penalty, length, area, observe, observe2, debug = sess.run(L_phi,feed_dict={img_tf: img_gray, phi_init: phi_old, img_g_tf: g,
            #                                                             gamma_pipe: g1, gamma_bg: g2, obs:out})
            penalty, length, area, observe2, debug = sess.run(L_phi,feed_dict={phi_init: phi_old})
            # if((iterations%100)==0):
            #     plt.title("length")
            #     plt.imshow(length)
            #     plt.show()

            # # all_energy = (coeff1 * penalty) + (coeff2 * length) + (coeff3 * area) + (coeff4 * observe) + (coeff5 * observe2)
            all_energy = (coeff1 * penalty) + (coeff2 * length) + (coeff3 * area) + (coeff5 * observe2)
            # all_energy = (coeff1 * penalty) + (coeff2 * length) + (coeff3 * area)
            phi_new = phi_old + (all_energy)
            phi_new = phi_new.astype('float32')
            phi_old = phi_new

            # print(phi_new)
            # plt.title("penalty")
            # plt.imshow(penalty)
            # plt.show()
            # print("a: %.5f, b: %.5f, c: %.5f, d: -, e: %.5f" % ((coeff1 * penalty).max(), (coeff2 * length).max(), (coeff3 * area).max(), (coeff5 * observe2).max()))
            """show result after iteration"""

            # plt.subplot(221)
            # plt.title("penalty tf")
            # plt.imshow(penalty)
            #
            # plt.subplot(222)
            # plt.title("length tf")
            # plt.imshow(length)
            #
            # plt.subplot(223)
            # plt.title("area tf")
            # plt.imshow(area)
            #
            # plt.subplot(224)
            # plt.title("obs tf")
            # plt.imshow(observe2)
            # plt.show()

            # plt.subplot(221)
            # plt.title("phi_tf")
            # plt.imshow(phi_old)
            # plt.subplot(222)
            # plt.title("debug_tf")
            # plt.imshow(debug)
            # plt.subplot(223)
            # plt.title("penalty_tf")
            # plt.imshow(penalty)
            # plt.show()

            # plt.title("debug_tf")
            # plt.imshow(debug)
            # plt.show()

            # print()

            plt.title("tensorflow")
            # plt.imshow(phi_old)
            plt.imshow(img_gray, cmap='gray')
            CS = plt.contour(phi_old, 0, colors='r', linewidths=2)
            plt.draw()
            if (i==iterations):
                plt.savefig(r"C:\Users\mook\PycharmProjects\LSM\experiment\021219(ver.4.3)\frame_" + str(index) + "_levelset.png")
            plt.show()
            plt.pause(0.05)
            plt.clf()
        plt.ioff()

    # return phi_old, joint_point, rect #out=suppress
    return phi_old