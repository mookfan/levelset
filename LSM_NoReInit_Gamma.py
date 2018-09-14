import numpy as np
import scipy.ndimage
import scipy.signal
import cv2
import matplotlib.pyplot as plt
import scipy.special as sc


def divergence(fx, fy):
    fyx, fxx = np.gradient(fx)
    fyy, fxy = np.gradient(fy)
    f = fxx + fyy
    return f

def deltafunction(epsilon, phi):
    k = (1.0 / (2.0 * epsilon)) * (1 + np.cos((np.pi * phi) / epsilon))
    delta = (0.0 * (np.abs(phi) > epsilon)) + (k * (np.abs(phi) <= epsilon))
    delta = delta.astype('float32')
    return delta


def energy(img, phi, epsilon, Ix, Iy):
    """INTERNAL____ENERGY"""
    # print phi.max()
    laplacian = cv2.Laplacian(phi, cv2.CV_32F)
    # Gx = cv2.Sobel(phi,cv2.CV_32F,1,0)
    # Gy = cv2.Sobel(phi,cv2.CV_32F,0,1)
    Gy, Gx = np.gradient(phi)
    magG = np.sqrt((Gx ** 2) + (Gy ** 2))
    plt.imshow(magG)
    plt.show()
    plt.clf()
    """Grad(phi)/mag(phi) ->> NORMALIZE !!!!!!!!!!!"""
    norm_Gx = Gx / (magG + (magG == 0))
    norm_Gy = Gy / (magG + (magG == 0))
    # plt.imshow(norm_Gx)
    # plt.show()
    # plt.imshow(norm_Gy)
    # plt.show()
    """div_sum_grad"""
    div = divergence(norm_Gx, norm_Gy)
    A = laplacian - div
    inner = A

    """EXTERNAL___ENERGY"""
    delta = deltafunction(epsilon, phi)
    # print("delta", delta.max(), delta.min())
    # !!!!! g !!!!!
    # print b
    # print "+++"
    edge = np.sqrt(np.square(Ix) + np.square(Iy))
    g = 1.0 / (1.0 + (edge ** 2))

    norm_Gx_length = g * norm_Gx
    norm_Gy_length = g * norm_Gy
    """div_sum_grad"""
    div = divergence(norm_Gx_length, norm_Gy_length)
    exter_length = delta * div

    # exter_area = v*delta
    exter_area = g * delta

    """for gray-scale"""
    # k = 2.0
    # theta = 2.0
    # """convert gray scale to binary>>> Actually it should use intensity graph"""
    # """??to avoid when too much intensity>>too less gamma pdf (near to very low intensity>> cannot separate these two apart"""
    # img = 1*(img == 255) #focus only 255 In this case>> image has two values >> 0 and 255
    # gamma_func = sc.factorial(k-1, exact=False)
    # gamma_pdf = (1.0/(gamma_func*(theta**k)))*(img**(k-1))*(np.exp(-(img/theta)))
    # # gamma_pdf = (0.0001*(gamma_pdf == 0)) + (gamma_pdf*(gamma_pdf > 0))
    # print("gamma_max: %.10f, gamma_min: %.10f" % (gamma_pdf.max(), gamma_pdf.min()))
    # # gp = gamma_pdf/gamma_pdf.max()
    # gp = (1*(gamma_pdf > 0.15))+(0.00001*(gamma_pdf == 0.0))
    # gn = (1*(gamma_pdf == 0.0))+(0.00001*(gamma_pdf > 0.15))
    # # print("gp:", gp)
    # # print("gn:", gn)
    # # print("gn/gp:", (np.log(gn/gp)))
    # # print("delta: " ,delta.max(), delta.min())
    # # print("result: ", (np.log(gn/gp))*delta)
    # gamma_energy = (np.log(gn/gp))*delta
    # print(gamma_energy.max(), gamma_energy.min())

    """Because intensity of this image has only two values so we can map from it's intensity to it's probability"""
    # gp = (1*(img == 255))+(0.00001*(img == 0))
    # gn = (1 * (img == 0)) + (0.00001 * (img == 255))

    gp = (1.0 * (img >= 100)) + (0.00001 * (img < 100))
    gn = (1.0 * (img < 100)) + (0.00001 * (img >= 100))
    # cv2.imshow("gp", gp)
    # cv2.imshow("gn", gn)
    # cv2.waitKey(1)
    gamma_energy = (np.log(gn / gp)) * delta

    debug = div
    return inner, exter_length, exter_area, gamma_energy, debug


def process(img, img_gray):
    init_phi = 6.0
    phi = (-1.0 * init_phi) * np.ones(img_gray.shape, 'float32')
    """for pipeline images"""
    # phi[100:300, 100:300] = phi[100:300, 100:300]+init_phi
    # phi[101:299, 101:299] = init_phi
    """some part of pipeline """
    phi[200:500, 300:600] = init_phi
    # phi[100:150, 100:250] = init_phi
    # phi[201:499, 301:599] = init_phi

    # phi[50:500, 300:600] = init_phi
    """ whole pipeline """
    # phi[200:600, 300:720] = phi[200:600, 300:720]+init_phi
    # phi[201:599, 301:719] = init_phi
    plt.imshow(phi)
    plt.show()

    mew = 0.12  # 0.12
    lamda = 50.0  # 5.0
    v = -5.0  # -5.0 #initial inside: -
    beta = 1.0
    epsilon = 1.0  # 1.5
    time = 0.5  # 0.5
    sigma = 1.0  # 1.0
    iter = 4000
    coeff1 = time * mew
    coeff2 = time * lamda
    coeff3 = time * v
    coeff4 = time * beta
    print("coeff1: %.2f, coeff2: %.2f, coeff3: %.2f" % (coeff1, coeff2, coeff3))

    # plt.imshow(img_gray)
    # plt.show()
    # print (img_gray.max())

    # """test gamma didtribution"""
    # k = 2.0
    # theta = 2.0
    # """convert gray scale to binary"""
    # img_gray = 1 * (img_gray == 255)
    # gamma_func = sc.factorial(k - 1, exact=False)
    # gamma_pdf = (1.0 / (gamma_func * (theta ** k))) * (img_gray ** (k - 1)) * (np.exp(-(img_gray / theta)))
    # # gamma_pdf = (0.0001*(gamma_pdf == 0)) + (gamma_pdf*(gamma_pdf > 0))
    # print("gamma_max: %.10f, gamma_min: %.10f" % (gamma_pdf.max(), gamma_pdf.min()))
    # # gp = gamma_pdf/gamma_pdf.max()
    # gp = (1 * (gamma_pdf > 0.15)) + (0.00001 * (gamma_pdf == 0.0))
    # gn = (1 * (gamma_pdf == 0.0)) + (0.00001 * (gamma_pdf > 0.15))
    # # print(gamma_pdf.shape)
    # gamma_energy = (np.log(gn / gp))
    # plt.imshow(gamma_energy)
    # plt.show()

    """smooth to denoise"""
    img_smooth = scipy.ndimage.filters.gaussian_filter(img_gray, sigma)
    plt.imshow(img_smooth)
    plt.show()

    """Edge_sobel"""
    Ix = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 0)
    Iy = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 1)

    plt.imshow(img_gray)
    CS = plt.contour(phi, 0, colors='g', linewidths=2)
    plt.draw()
    plt.show()
    plt.ion()

    for i in range(0, iter):
        print("times#: ", i)
        print("phi_max: %f, phi_min: %f" % (phi.max(), phi.min()))
        internal, external_length, external_area, gamma_energy,  debug = energy(img_gray, phi, epsilon, Ix, Iy)
        plt.imshow(debug)
        plt.show()
        # print("part1: %.2f, part2: %.2f, part3: %.2f, part4: %.2f" % (internal, external_length, external_area, gamma_energy))
        """-gamma_energy because pipe area returns negative energy BUT phi is determined positive inside the pipe. """
        phi = phi + (coeff1 * internal) + (coeff2 * external_length) + (coeff3 * external_area) - (coeff4 * gamma_energy)
        phi = phi.astype('float32')
        print("a: %.2f, b: %.2f, c: %.2f, d: %.2f" % ((coeff1 * internal).max(), (coeff2 * external_length).max(), (coeff3 * external_area).max(), (beta * gamma_energy).max()))
        # if i % 1 == 0:
        #     plt.imshow(internal)
        #     plt.show()
        #     plt.pause(0.05)
        #     plt.clf()
        if i % 10 == 0:
            plt.imshow(phi)
            CS = plt.contour(phi, 0, colors='r', linewidths=2)
            plt.draw()
            plt.show()
            plt.pause(0.05)
            plt.clf()
        if i % 100 == 0:
            plt.imshow(img_gray)
            CS = plt.contour(phi, 0, colors='r', linewidths=2)
            plt.draw()
            plt.show()
            plt.pause(0.05)
            plt.clf()
        print()
    plt.ioff()
    plt.imshow(img_gray)
    CS = plt.contour(phi, 0, colors='g', linewidths=2)
    plt.draw()
    plt.show()


if __name__ == "__main__":
    # img = cv2.imread('original.png')
    img = cv2.imread('Feature.png')
    print(img.shape)
    # img = cv2.imread('image_RTheta.jpg')
    # img = cv2.imread('fig2.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    process(img, img_gray)