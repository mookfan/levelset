import cv2
import numpy as np
import tensorflow as tf
import scipy.ndimage
import matplotlib.pyplot as plt



# def magnitude(phi_tf):
#     rows, cols = img.shape
#     a = tf.reshape(phi_tf, shape=[-1, rows, cols, 1])
#     """gradient"""
#     x_weight = tf.reshape(tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], tf.float32), shape=[3, 3, 1, 1])
#     y_weight = tf.reshape(tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], tf.float32), shape=[3, 3, 1, 1])
#     grad_x = tf.nn.conv2d(a, x_weight, strides=[1, 1, 1, 1], padding='SAME')
#     grad_y = tf.nn.conv2d(a, y_weight, strides=[1, 1, 1, 1], padding='SAME')
#     # grad_x = tf.reshape(grad_x, shape=[rows, cols])
#     # grad_y = tf.reshape(grad_y, shape=[rows, cols])
#     # magnitude gradient
#     mag = tf.sqrt(tf.pow(grad_x, 2) + tf.pow(grad_y, 2))
#     mag = tf.reshape(mag, shape=[rows, cols])
#     # print(mag)
#     return mag

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

def observe_pipeline(img, rows, cols):
    thres = 100
    thres_tf = tf.constant(thres*np.ones([rows, cols]), tf.float32)
    cond = tf.greater(img, thres_tf)
    out = tf.where(cond, tf.ones([rows, cols]), 0.00001*tf.ones([rows, cols]))
    return out

def observe_no_pipeline(img, rows, cols):
    thres = 100
    thres_tf = tf.constant(thres*np.ones([rows, cols]), tf.float32)
    cond = tf.greater(img, thres_tf)
    out = tf.where(cond, 0.00001*tf.ones([rows, cols]), tf.ones([rows, cols]))
    return out

def energy(img, phi, img_g_tf):
    epsilon = 1.0
    rows, cols = img.shape
    # phi_ = tf.Variable(tf.ones([rows, cols]), tf.float32)
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

    """PROBLEM!!!!!"""
    gp = observe_pipeline(img, rows, cols)
    gn = observe_no_pipeline(img, rows, cols)
    part = tf.divide(gn, gp)
    log = tf.log(part)
    observe = tf.multiply(delta, log)

    debug = inner
    return inner, exter_length, exter_area, observe, debug

img = cv2.imread("Feature.png")
# img = cv2.imread(r"D:\Mook\LevelSet\Images\multiLook_10.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img_gray + (img_gray == 0)

init_phi = 6.0
phi_first = (-1.0 * init_phi) * np.ones(img_gray.shape, 'float32')
phi_first[200:500, 300:600] = init_phi
# phi_first[100:150, 100:250] = init_phi
# phi_first = phi_first.astype('float32')
# print(phi_first.shape, phi_first.dtype)
plt.imshow(img_gray)
plt.show()

mew = 0.12  # 0.12
lamda = 50.0  # 5.0
v = -5.0  # -5.0 #initial inside: -
beta = -1.0
epsilon = 1.0  # 1.5
time = 0.5  # 0.5
sigma = 1.0  # 1.0
iterations = 1000
coeff1 = time * mew
coeff2 = time * lamda
coeff3 = time * v
coeff4 = time * beta
print("coeff1: %.2f, coeff2: %.2f, coeff3: %.2f" % (coeff1, coeff2, coeff3))

img_smooth = scipy.ndimage.filters.gaussian_filter(img_gray, sigma)
plt.imshow(img_smooth)
plt.show()

Ix = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 0)
Iy = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 1)

edge = np.sqrt(np.square(Ix) + np.square(Iy))
g = 1.0 / (1.0 + (edge ** 2))
# plt.imshow(g)
# plt.show()
rows, cols = img_gray.shape

# input = tf.placeholder("float32")
input = tf.Variable(phi_first, dtype=tf.float32)
# mag_tf = tf.placeholder("float32")

img_tf = tf.Variable(img_smooth,  dtype=tf.float32)
img_g_tf = tf.Variable(g, dtype=tf.float32)

# mag = magnitude(input)

L_phi = energy(img_tf, input, img_g_tf)


phi_old = phi_first

# plt.imshow(phi_old)
# CS = plt.contour(phi_old, 0, colors='r', linewidths=2)
# plt.draw()
# plt.show()
# plt.ion()
with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    for i in range(iterations+1):
        print("times#: ", i)
        # img = sess.run(img_tf)
        # print("phi_max: %f, phi_min: %f" % (phi_old.max(), phi_old.min()))
        # print("img: ", img.max())
        penalty, length, area, observe, debug = sess.run(L_phi, feed_dict={img_tf: img_smooth, input: phi_old, img_g_tf: g})
        all_energy = (coeff1 * penalty) + (coeff2 * length) + (coeff3 * area) + (coeff4 * observe)
        phi_new = phi_old + (all_energy)
        phi_new = phi_new.astype('float32')
        phi_old = phi_new
        # print(phi_new)
        print("a: %.2f, b: %.2f, c: %.2f, d: %.2f" % ((coeff1 * penalty).max(), (coeff2 * length).max(), (coeff3 * area).max(),(coeff4 * observe).max()))
        if i % 500 == 0:
            plt.imshow(img_smooth)
            CS = plt.contour(phi_old, 0, colors='r', linewidths=2)
            plt.draw()
            plt.show()
            plt.pause(0.05)
            plt.clf()
        print()
        # plt.ioff()