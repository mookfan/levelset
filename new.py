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
    cond = tf.greater(ab_phi, eps)

    param1 = tf.constant((1.0/(2.0*epsilon)) * np.ones([rows, cols]), tf.float32)
    coef = tf.multiply(pi_eps, phi_tf)
    param_ = tf.cos(coef)
    param2 = tf.multiply(param1, param_)
    var = tf.add(param1, param2)

    out = tf.where(cond, tf.zeros([rows, cols]), var)
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
    coeff1 = 0.12  # 0.12
    coeff2 = 50.0  # 5.0
    coeff3 = -5.0  # -5.0 #initial inside: -
    beta = 1.0
    rows, cols = img.shape
    phi_ = tf.Variable(tf.ones([rows, cols]), tf.float32)
    # phi_tf = tf.add(offset, phi)
    phi_tf = tf.multiply(phi, phi_)
    a = tf.reshape(phi_tf, shape=[-1, rows, cols, 1])
    """laplacian"""
    filter_laplacian = tf.reshape(tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], tf.float32), shape=[3, 3, 1, 1])
    lap = tf.nn.conv2d(a, filter_laplacian, strides=[1, 1, 1, 1], padding="SAME")
    lap = tf.reshape(lap, shape=[rows, cols])
    """gradient"""
    x_weight = tf.reshape(tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], tf.float32), shape=[3, 3, 1, 1])
    y_weight = tf.reshape(tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], tf.float32), shape=[3, 3, 1, 1])
    grad_x = tf.nn.conv2d(a, x_weight, strides=[1, 1, 1, 1], padding='SAME')
    grad_y = tf.nn.conv2d(a, y_weight, strides=[1, 1, 1, 1], padding='SAME')
    grad_x_re = tf.reshape(grad_x, shape=[rows, cols])
    grad_y_re = tf.reshape(grad_y, shape=[rows, cols])
    # magnitude gradient
    mag_tf = tf.sqrt(tf.pow(grad_x_re, 2)+tf.pow(grad_y_re, 2))
    thres = tf.constant(np.zeros([rows, cols]), tf.float32)
    cond = tf.equal(mag_tf, thres)
    mag = tf.where(cond, tf.ones([rows, cols]), mag_tf)
    # print(grad_y)
    mag = tf.reshape(mag, shape=[-1, rows, cols, 1])
    norm_x = tf.div(grad_x, mag)
    norm_y = tf.div(grad_y, mag)
    fx = tf.nn.conv2d(norm_x, x_weight, strides=[1, 1, 1, 1], padding='SAME')
    fy = tf.nn.conv2d(norm_y, y_weight, strides=[1, 1, 1, 1], padding='SAME')
    fx = tf.reshape(fx, shape=[rows, cols])
    fy = tf.reshape(fy, shape=[rows, cols])
    div1 = tf.add(fx, fy)
    inner = tf.subtract(lap, div1)
    penalty = tf.scalar_mul(coeff1, inner)


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
    length = tf.scalar_mul(coeff2, exter_length)

    exter_area = tf.multiply(img_g_tf, delta)
    area = tf.scalar_mul(coeff3, exter_area)

    """PROBLEM!!!!!"""
    gp = observe_pipeline(img, rows, cols)
    gn = observe_no_pipeline(img, rows, cols)
    part = tf.div(gn, gp)
    observe = tf.scalar_mul(beta, tf.log(part))

    allenergy = tf.add(tf.add(tf.add(penalty, length), area), observe)

    return allenergy

time = 0.5  # 0.5
sigma = 1.0  # 1.0
iterations = 2000
img = cv2.imread('Feature2.png', 0)
img_smooth = scipy.ndimage.filters.gaussian_filter(img, sigma)
Ix = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 0)
Iy = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 1)
edge = np.sqrt(np.square(Ix) + np.square(Iy))
g = 1.0 / (1.0 + (edge ** 2))
rows, cols = img.shape
init_phi = 6.0

phi_first = (-1.0 * init_phi) * np.ones((rows, cols), 'float32')
# phi_first[200:500, 300:600] = init_phi
phi_first[100:150, 100:250] = init_phi
phi_first = phi_first.astype('float32')
# print(phi_first.shape, phi_first.dtype)

input = tf.placeholder("float32")
# mag_tf = tf.placeholder("float32")

img_tf = tf.constant(img_smooth, tf.float32)
img_g_tf = tf.constant(g, tf.float32)

# mag = magnitude(input)

L_phi = energy(img_tf, input, img_g_tf)


phi_old = phi_first
phi_new = np.zeros([rows, cols])

# plt.imshow(phi_old)
# CS = plt.contour(phi_old, 0, colors='r', linewidths=2)
# plt.draw()
# plt.show()
with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    for i in range(iterations):
        print("times#: ", i)
        all_energy = sess.run(L_phi, feed_dict={input: phi_old})
        phi_new = phi_old + (time*all_energy)
        phi_old = phi_new
        # print(phi_new)
        if i % 1 == 0:
            plt.imshow(img_smooth)
            CS = plt.contour(phi_old, 0, colors='r', linewidths=2)
            plt.draw()
            plt.show()
            plt.pause(0.05)
            plt.clf()