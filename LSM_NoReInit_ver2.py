import numpy as np
import scipy.ndimage
import scipy.signal
import cv2
import matplotlib.pyplot as plt

def divergence(fx, fy):
    fyx, fxx = np.gradient(fx)
    fyy, fxy = np.gradient(fy)
    f = fxx + fyy
    return f

def deltafunction(epsilon, phi):
    k = (1.0/(2.0*epsilon))*(1+np.cos((np.pi*phi)/epsilon))
    delta = (0.0*(np.abs(phi)>epsilon))+(k*(np.abs(phi)<=epsilon))
    delta = delta.astype('float32')
    return delta

def energy(img, phi, epsilon, Ix, Iy):
    """INTERNAL____ENERGY"""
    # print phi.max()
    laplacian = cv2.Laplacian(phi,cv2.CV_32F)
    # Gx = cv2.Sobel(phi,cv2.CV_32F,1,0)
    # Gy = cv2.Sobel(phi,cv2.CV_32F,0,1)
    Gy, Gx = np.gradient(phi)
    magG = np.sqrt((Gx**2)+(Gy**2))
    plt.imshow(magG)
    plt.show()
    plt.clf()
    """Grad(phi)/mag(phi) ->> NORMALIZE !!!!!!!!!!!"""
    norm_Gx = Gx/(magG+(magG==0))
    norm_Gy = Gy/(magG+(magG==0))
    # plt.imshow(norm_Gx)
    # plt.show()
    # plt.imshow(norm_Gy)
    # plt.show()
    """div_sum_grad"""
    div = divergence(norm_Gx,norm_Gy)
    A = laplacian-div
    inner = A

    """EXTERNAL___ENERGY"""
    delta = deltafunction(epsilon, phi)
    #!!!!! g !!!!!
    # print b
    # print "+++"
    edge = np.sqrt(np.square(Ix)+np.square(Iy))
    g = 1.0/(1.0+(edge**2))

    norm_Gx_length = g*norm_Gx
    norm_Gy_length = g*norm_Gy
    """div_sum_grad"""
    div = divergence(norm_Gx_length,norm_Gy_length)    
    exter_length = delta*div

    # exter_area = v*delta
    exter_area = g*delta 


    return inner, exter_length, exter_area
   

def process(img, img_gray):
    init_phi = 6.0
    phi = (-1.0*init_phi)*np.ones(img_gray.shape,'float32')
    """for pipeline images"""
    # phi[100:300, 100:300] = phi[100:300, 100:300]+init_phi
    # phi[101:299, 101:299] = init_phi
    """some part of pipeline """
    phi[200:500, 300:600] = init_phi
    # phi[201:499, 301:599] = init_phi
    """ whole pipeline """
    # phi[200:600, 300:720] = phi[200:600, 300:720]+init_phi
    # phi[201:599, 301:719] = init_phi
    plt.imshow(phi)
    plt.show()

    mew = 0.12 #0.12
    lamda = 50.0 #5.0 
    v = -5.0 #-5.0 #initial inside: - 
    epsilon = 1.5 #1.5
    time = 0.5 #0.5
    sigma = 1.0 #1.0
    iter = 5000
    coeff1 = time*mew
    coeff2 = time*lamda
    coeff3 = time*v
    print ("coeff1: %.2f, coeff2: %.2f, coeff3: %.2f" %(coeff1, coeff2, coeff3))

    """smooth to denoise"""
    img_smooth = scipy.ndimage.filters.gaussian_filter(img_gray, sigma)
    plt.imshow(img_smooth)
    plt.show()

    """Edge_sobel"""
    Ix = cv2.Sobel(img_smooth,cv2.CV_32F,1,0)
    Iy = cv2.Sobel(img_smooth,cv2.CV_32F,0,1)

    plt.imshow(img_gray)
    CS = plt.contour(phi,0, colors='g' ,linewidths=2)
    plt.draw()
    plt.show()
    plt.ion()

    for i in range(0, iter):
        print ("times#: ", i)
        print ("phi_max: %f, phi_min: %f" %(phi.max(), phi.min()))
        internal, external_length, external_area = energy(img_gray, phi, epsilon, Ix, Iy) 
        phi = phi+(coeff1*internal)+(coeff2*external_length)+(coeff3*external_area)
        phi = phi.astype('float32')
        if i % 10==0:
            plt.imshow(phi)
            CS = plt.contour(phi,0, colors='r' ,linewidths=2)
            plt.draw()
            plt.show()
            plt.pause(0.05)
            plt.clf()
        if i % 100==0:
            plt.imshow(img_gray)
            CS = plt.contour(phi,0, colors='r' ,linewidths=2)
            plt.draw()
            plt.show()
            plt.pause(0.05)
            plt.clf()
    plt.ioff()
    plt.imshow(img_gray)
    CS = plt.contour(phi,0, colors='g' ,linewidths=2)
    plt.draw()
    plt.show()

if __name__ == "__main__":
    img = cv2.imread('Feature1.png')
    # img = cv2.imread('image_RTheta.jpg')
    # img = cv2.imread('fig2.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    process(img, img_gray)