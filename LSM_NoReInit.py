import numpy as np
import scipy.ndimage
import scipy.signal
import cv2
import matplotlib.pyplot as plt
import numdifftools as nd


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

def energy(img,mew, phi, lamda, v, epsilon, Ix, Iy):
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
    inner = mew*A

    """EXTERNAL___ENERGY"""
    delta = deltafunction(epsilon, phi)
    #!!!!! g !!!!!
    # print b
    # print "+++"
    edge = np.sqrt(np.square(Ix)+np.square(Iy))
    g = 1.0/(1.0+(edge**2))


    """!!!!!!!THIS MAKE THE RESULT AS RIPPLES!!!!!!!!!!!!"""
    norm_Gx_length = g*norm_Gx
    norm_Gy_length = g*norm_Gy
    """div_sum_grad"""
    div = divergence(norm_Gx_length,norm_Gy_length)    
    exter_length = lamda*delta*div

    # exter_area = v*delta
    exter_area = v*g*delta 

    """Pball's part' workjaaaa"""
    mean_value = 200.0
    cov_value = 20.0
    E = (((img-mean_value)**2)/(cov_value))+np.log(cov_value)
    (row,col) = np.nonzero(E<cov_value)
    E[row,col] = -mean_value
    E4 = (delta*E).astype('float32')
    # """mook's part NOT work'"""
    # E = (1.0/(1.0+(g*(edge**2))))*delta
    # (row,col) = np.nonzero(E<cov_value)

    return inner, exter_length, exter_area, E4
    

def process(img, img_gray):
    coeff_phi = 2
    phi = (-1*coeff_phi)*np.ones(img_gray.shape,'float32')
    # """for circle image (center)"""
    phi[100:101, :] = coeff_phi
    # phi[50:100, 50:100] = coeff_phi
    # """for circle image (buttom-right)"""
    # phi[200:230, 200:240] = coeff_phi
    """for pipeline images"""
    # phi[100:200, 100:200] = coeff_phi
    # phi[348:360, 440:525] = coeff_phi

    mew = 0.01 #0.01
    lamda = 0.05 #0.005 
    v = 0.005 #-0.005 #initial inside: -
    epsilon = 2.0 #2.0
    time = 1.0 #1.0
    sigma = 1 #1
    iter = 1000
    beta = 1.0 #1.0
    """smooth to denoise"""
    img_smooth = scipy.ndimage.filters.gaussian_filter(img_gray, sigma)
    plt.imshow(img_smooth)
    plt.show()
    # """No smooth: cannot get circle's boundary'"""
    # img_smooth = img_gray
    """Edge_sobel"""
    Ix = cv2.Sobel(img_smooth,cv2.CV_32F,1,0)
    Iy = cv2.Sobel(img_smooth,cv2.CV_32F,0,1)
    # edge = np.sqrt(np.square(Ix)+np.square(Iy))
    # g = 1.0/(1.0+(edge**2))
    # plt.imshow(g)
    # plt.show()
    """Edge_gradient"""
    # Iy, Ix = np.gradient(img_smooth)

    # plt.imshow(Ix)
    # plt.show()
    # plt.imshow(Iy)
    # plt.show()
    # internal, external_length, external_area, E = energy(mew, phi, lamda, v, epsilon, Ix, Iy) 
    print "whileloop"

    plt.imshow(img_gray)
    # plt.hold(True)
    CS = plt.contour(phi,0, colors='g' ,linewidths=2)
    plt.draw()
    # time.sleep(0.1)
    # plt.hold(False)
    plt.show()
    plt.ion()
    # print "++++++++++++"
    # print phi.shape
    for i in range(0, iter):
        print "times#: ", i
        internal, external_length, external_area, E = energy(img_gray,mew, phi, lamda, v, epsilon, Ix, Iy) 
        # print "vvvvvvvvvvvvvv"
        # print phi.shape
        # print internal.shape, external_length.shape, external_area.shape
        phi = phi+(time*(internal+external_length+external_area-(beta*E)))
        # phi = phi+(time*(internal+external_length-external_area))
        phi = phi.astype('float32')
        # print "________________________________"
        # print phi.shape
        if i % 10==0:
            plt.imshow(phi)
            CS = plt.contour(phi,0, colors='r' ,linewidths=2)
            plt.draw()
            plt.show()
            plt.pause(0.05)
            plt.clf()
    plt.ioff()
    plt.imshow(img_gray)
    # plt.hold(True)
    CS = plt.contour(phi,0, colors='g' ,linewidths=2)
    plt.draw()
    # time.sleep(0.1)
    # plt.hold(False)
    plt.show()

if __name__ == "__main__":
    # img = cv2.imread('Feature.png')
    # img = cv2.imread('image_RTheta.jpg')
    img = cv2.imread('fig2.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    process(img, img_gray)
    # print img.dtype

