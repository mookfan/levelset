import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
from skimage import color, io
import cv2
import pickle as pl
from skimage.morphology import skeletonize_3d

def grad(x):
    return np.array(np.gradient(x))


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def stopping_fun(x):
    return 1. / (1. + norm(grad(x))**2)


def curvature(f):
    fy, fx = grad(f)
    norm = np.sqrt(fx**2 + fy**2)
    Nx = fx / (norm + 1e-8)
    Ny = fy / (norm + 1e-8)
    return div(Nx, Ny)


def div(fx, fy):
    fyy, fyx = grad(fy)
    fxy, fxx = grad(fx)
    return fxx + fyy


def dot(x, y, axis=0):
    return np.sum(x * y, axis=axis)

def cafar(all_size, win_size, img_gray):
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
    return ratio, testing_cell


# img = cv2.imread(r"C:\Users\mook\Desktop\lsm_ori.png", 0)
img = cv2.imread(r"C:\Users\mook\PycharmProjects\LSM\images\multilook.png", 0)
img = img[:490, 10:758]
img = img - np.mean(img)

# plt.imshow(img)
# plt.show()

# Smooth the image to reduce noise and separation between noise and edge becomes clear
img_smooth = img = cv2.GaussianBlur(img, (9, 9), 3)
img_smooth = cv2.normalize(img_smooth, None, 0, 255, cv2.NORM_MINMAX)
img_smooth = (1*(img_smooth<=1))+(img_smooth*(img_smooth>1))
print(img_smooth.max(), img_smooth.min())
plt.imshow(img_smooth)
plt.show()


v = 10.
dt = 1.
mew = 5.
alpha = 1.

# g = stopping_fun(img_smooth)
# dg = grad(g)

row, col = img.shape
phi_coeff = 1
phi = (-phi_coeff) * np.ones((row, col), 'float32')
phi[:300, :] = phi_coeff

# plt.imshow(phi)
# plt.show()


CAFAR, testing_cell = cafar(51, 41, img_smooth)
CAFAR = (0.1 * (CAFAR < 0.1)) + (CAFAR * (CAFAR >= 0.1))
print("CAFAR:", CAFAR.max(), CAFAR.min())
g = 1/CAFAR
# g = (0*(g<=0.5))+(g*(g>0.5))
# plt.subplot(121)
# plt.title("CAFAR")
# plt.imshow(CAFAR)
# plt.subplot(122)
# plt.title("g")
# plt.imshow(g)
# plt.show()
# """centerline"""
# mask = (1*(CAFAR>=1.6))+(0*(CAFAR<1.6))
# mask = skeletonize_3d(mask)
# # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
# # mask = cv2.dilate(mask, kernel, iterations=1)
# mask = (1*(mask==0))+((0)*(mask>0))
# plt.title("mask for centerline")
# plt.imshow(mask)
# plt.show()
#
dg = grad(g)
g1, g2 = np.gradient(g)
gg = g1+g2
plt.imshow(gg)
plt.show()



# f = lambda x,y, x0,y0, sig: np.exp((-(x-x0)**2- (y-y0)**2)/sig**2)
X,Y = np.meshgrid(np.arange(0, col, 1), np.arange(0, row, 1))
# array = f(X,Y, 24,24,7.)

dy, dx = np.gradient(g)
n = 7
fig = plt.figure()
plt.imshow(img_smooth, cmap='gray')
plt.quiver(X[::n,::n],Y[::n,::n],dx[::n,::n], dy[::n,::n],
           np.sqrt(dx[::n,::n]**2+dy[::n,::n]**2),  units="xy", scale=0.04, cmap="Reds")
pl.dump(fig, open(r"C:\Users\mook\PycharmProjects\LSM\experiment\results\testtttt.pickle",'wb'))
plt.show()


plt.ion()
for i in range(1000):
    print(i)
    dphi = grad(phi)
    dphi_norm = norm(dphi)
    kappa = curvature(phi)

    smoothing = g * kappa * dphi_norm *mew
    balloon = g * dphi_norm * v
    attachment = dot(dphi, dg) * alpha

    dphi_t = smoothing + balloon + attachment

    phi = phi + dt * dphi_t

    # plt.subplot((221))
    # plt.imshow(img_smooth, cmap='gray')
    # term12, term11 = np.gradient(smoothing)
    # plt.quiver(X[::n, ::n], Y[::n, ::n], term11[::n, ::n], term12[::n, ::n],
    #            np.sqrt(term11[::n, ::n] ** 2 + term12[::n, ::n] ** 2), units="xy", scale=0.04, cmap="Reds")
    # plt.subplot((222))
    # plt.imshow(img_smooth, cmap='gray')
    # term22, term21 = np.gradient(balloon)
    # plt.quiver(X[::n, ::n], Y[::n, ::n], term21[::n, ::n], term22[::n, ::n],
    #            np.sqrt(term21[::n, ::n] ** 2 + term22[::n, ::n] ** 2), units="xy", scale=0.04, cmap="Reds")
    # plt.subplot((223))
    # plt.imshow(img_smooth, cmap='gray')
    # term32, term31 = np.gradient(attachment)
    # plt.quiver(X[::n, ::n], Y[::n, ::n], term31[::n, ::n], term32[::n, ::n],
    #            np.sqrt(term31[::n, ::n] ** 2 + term32[::n, ::n] ** 2), units="xy", scale=0.04, cmap="Reds")
    # plt.subplot((224))
    plt.imshow(phi)
    plt.imshow(img, cmap='gray')
    CS = plt.contour(phi, 0, colors='r', linewidths=2)
    plt.draw()
    plt.show()
    plt.pause(5.0)
    plt.clf()
plt.ioff()