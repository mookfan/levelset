import cv2
import numpy as np
import tensorflow as tf
import scipy.ndimage
import matplotlib.pyplot as plt

class levelset:
    def __init__(self, Img):
        self.init_phi = 6.0
        self.mew = 0.12  # 0.12
        self.lamda = 50.0  # 5.0
        self.v = -5.0  # -5.0 #initial inside: -
        self.beta = 1.0
        self.epsilon = 1.0  # 1.5
        self.time = 0.5  # 0.5
        self.iterations = 3000

        self.rows, self.cols = Img.shape
        self.img = tf.placeholder(tf.float32, shape=(self.rows, self.cols))
        self.param_tf = tf.Variable(tf.zeros([self.rows, self.cols]), tf.float32, name='param_tf')
        self.phi = np.zeros([self.rows, self.cols])


    def divergence(self, fx, fy):
        fyx, fxx = np.gradient(fx)
        fyy, fxy = np.gradient(fy)
        f = fxx + fyy
        return f

    def deltafunction(self):
        k = (1.0 / (2.0 * self.epsilon)) * (1 + np.cos((np.pi * self.phi) / self.epsilon))
        delta = (0.0 * (np.abs(self.phi) > self.epsilon)) + (k * (np.abs(self.phi) <= self.epsilon))
        delta = delta.astype('float32')
        return delta

    def energy_cal(self, grad_x_init, grad_y_init, img_smooth):
        """1. INTERNAL____ENERGY"""
        laplacian = cv2.Laplacian(self.phi, cv2.CV_32F)
        grad_y, grad_x = np.gradient(self.phi)
        mag_grad = np.sqrt((grad_y ** 2) + (grad_x ** 2))
        # magni_tf = tf.sqrt(tf.pow(self.param_tf, 2) + tf.pow(self.param_tf, 2))
        # mag_grad = tf.Session().run(magni_tf, feed_dict={self.param_tf: grad_y, self.param_tf: grad_x})
        # plt.imshow(mag_grad)
        # plt.show()
        norm_Gx = grad_x / (mag_grad + (mag_grad == 0))
        norm_Gy = grad_y / (mag_grad + (mag_grad == 0))
        div_penal = self.divergence(norm_Gx, norm_Gy)
        # print(div_penal.shape)
        internal = laplacian - div_penal

        """2. LENGTH____ENERGY"""
        edge = np.sqrt((grad_y_init ** 2) + (grad_x_init ** 2))
        # edge_tf = tf.sqrt(tf.pow(self.param_tf, 2) + tf.pow(self.param_tf, 2))
        # edge = tf.Session().run(edge_tf, feed_dict={self.param_tf: grad_y_init, self.param_tf: grad_x_init})
        g = 1.0 / (1.0 + (edge ** 2))
        delta = self.deltafunction()
        div_length = self.divergence(g * norm_Gx, g * norm_Gy)
        # print("div_length", div_length.max(), div_length.min())
        # print("delta", delta.max(), delta.min())
        external_length = delta * div_length

        """3. AREA____ENERGY"""
        external_area = g * delta

        """4. OBSERVATION____ENERGY"""
        gp = (1.0 * (img_smooth >= 100)) + (0.00001 * (img_smooth < 100))
        gn = (1.0 * (img_smooth < 100)) + (0.00001 * (img_smooth >= 100))
        observation_energy = (np.log(gn / gp)) * delta
        return internal, external_length, external_area, observation_energy
        # return 0, 0, 0, 0

    def process(self, img):
        img_smooth = scipy.ndimage.filters.gaussian_filter(Img, sigma)
        grad_x_init = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 0)
        grad_y_init = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 1)

        coeff1 = self.time * self.mew
        coeff2 = self.time * self.lamda
        coeff3 = self.time * self.v
        phi_init_tf = tf.Variable(-self.init_phi * tf.ones([self.rows, self.cols]), tf.float32, name='phi_init_tf')

        model = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(model)
            self.phi = session.run(phi_init_tf)
            self.phi[200:500, 300:600] = 6.0
            self.phi = session.run(phi_init_tf, feed_dict={phi_init_tf: self.phi})
            # plt.imshow(phi)
            # plt.show()
            for i in range(0, self.iterations):
                print("times#: ", i)
                print("phi_max: %f, phi_min: %f" % (self.phi.max(), self.phi.min()))
                internal, external_length, external_area, observation_energy = self.energy_cal(grad_x_init, grad_y_init, img_smooth)
                self.phi = self.phi + (coeff1 * internal) + (coeff2 * external_length) + (coeff3 * external_area) - (self.beta * observation_energy)
                print("a: %.2f, b: %.2f, c: %.2f, d: %.2f" % ((coeff1 * internal).max(), (coeff2 * external_length).max(), (coeff3 * external_area).max(),(self.beta * observation_energy).max()))
                self.phi = self.phi.astype('float32')
                if i % 100 == 0:
                    plt.imshow(img_smooth)
                    CS = plt.contour(self.phi, 0, colors='r', linewidths=2)
                    plt.draw()
                    plt.show()
                    plt.pause(0.05)
                    plt.clf()
                print()

if __name__ == "__main__":
    sigma = 1.0  # 1.0
    Img = cv2.imread('Feature3.png', 0)
    proc = levelset(Img)
    proc.process(Img)