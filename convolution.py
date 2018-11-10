import cv2
import numpy as np

if __name__ == "__main__":

    st = open("week10_Q8_image.npy", 'r')
    img = np.loadtxt(st, delimiter=' ')
    convo = np.array([[0.053, 0.110, 0.54], [0.111, 0.225, 0.111], [0.054, 0.110, 0.053]])

    s, n, p = 1, 3, 1
    """
        s is stride 
        n is the size of kernel
        p is the padding
    """
    res = np.zeros((28 - n + 1 + 2 * p, 28 - n + 1 + 2 * p))
    paddedImg = np.zeros((28 + 2 * p, 28 + 2 * p))

    paddedImg[p:28+p, p:28+p] = img

    for i in range(0, 28 - n + 1 + 2 * p, s):
        for j in range(0, 28 - n + 1 + 2 * p, s):
            res[i][j] = np.sum(convo * paddedImg[i:i + n, j:j + n])

    cv2.imshow("original", img)
    cv2.imshow("changed", res)
    cv2.waitKey()
    cv2.destroyAllWindows()