import numpy as np
import cv2
import matplotlib.pyplot as plt

# def createCurve():
#     x = np.linspace(-2, 2, 100)
#     print x
#     y = 1.0/(1.0+np.exp(-5.0*x))
#     plt.figure()
#     plt.plot(x, y, lw = 10)
#     plt.xlabel('$x$')
#     plt.ylabel('$\exp(x)$')
#     plt.show()

if __name__ == '__main__':
    # createCurve()
    img = cv2.imread('Feature.png')
    cv2.imshow('img', img)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()