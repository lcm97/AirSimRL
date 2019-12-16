import numpy
import math
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt

class Sobel:
    """
    Sobel algorithm version 2
    WIKIPEDIA
    The Sobel operator, sometimes called the Sobel–Feldman operator or Sobel filter,
    is used in image processing and computer vision, particularly within edge detection
     algorithms where it creates an image emphasising edges.
     The operator uses two 3×3 kernels which are convolved with the original image to
     calculate approximations of the derivatives – one for horizontal changes, and one for vertical.
     If we define A as the source image, and Gx and Gy are two images which at each point contain
     the horizontal and vertical derivative approximations respectively, the computations are as follows:
            |+1 0 -1|               |+1 +2 +1|
     Gx =   |+2 0 -2| * A  and Gy = | 0  0  0| * A
            |+1 0 -1|               |-1 -2 -1|
     Where * here denotes the 2-dimensional signal processing convolution operation
     Since the Sobel kernels can be decomposed as the products of an averaging and a differentiation kernel,
     they compute the gradient with smoothing. For example,Gx can be written as
        |+1 0 -1|     |1|
        |+2 0 -2|  =  |2| [+1 0 -1]
        |+1 0 -1|     |1|
     G = sqrt(Gx ** 2 + Gy **2)
     gradient direction
     alpha = atan(Gy/Gx
     """

    def __init__(self, array_):

        # kernel flipped for the convolution
        self.gx = numpy.array(([-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]))
        # kernel flipped
        self.gy = numpy.array(([-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]))
        self.kernel_half = 1
        self.shape = array_.shape
        self.array = array_
        self.source_array = numpy.zeros((self.shape[0], self.shape[1], 3))
        self.threshold = 0

    def run(self):

        # Starting at row 1, finishing at shape[0] - 1 due to the size of the kernel
        # and to avoid IndexError
        for y in range(2, self.shape[1]-2):

            for x in range(2, self.shape[0]-2):
                # Apply both kernels at once for each pixels
                # Horizontal kernel Gx
                data = self.array[x - 1:x + 2, y - 1:y + 2][:, :, 0]
                s1 = sum(sum(numpy.multiply(data, self.gx)))
                # Vertical kernel Gy
                s2 = sum(sum(numpy.multiply(data, self.gy)))
                magnitude = math.sqrt(s1 ** 2 + s2 ** 2)
                # update the pixel if the magnitude is above threshold else black pixel
                self.source_array[x, y] = magnitude if magnitude > self.threshold else 0
        # cap the values
        numpy.putmask(self.source_array, self.source_array > 255, 255)
        numpy.putmask(self.source_array, self.source_array < 0, 0)
        return self.source_array


if __name__ == '__main__':

    img = cv2.imread("./images2/img_428.png")
    npix = img.shape[0] * img.shape[1]
    result = gaussian_filter(img, sigma=1)

    Sob = Sobel(result)
    img_sob = Sob.run()
    print(numpy.mean(img_sob))
    print(numpy.sqrt(1.0 / npix * numpy.sum(img_sob * img_sob)))
    print(numpy.std( img_sob ))

    plt.imshow(img_sob.astype('uint8'))
    plt.show()
