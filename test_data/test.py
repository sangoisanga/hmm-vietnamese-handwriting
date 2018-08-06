import time
import cv2
import numpy as np


if __name__ == '__main__':
    img = cv2.imread('A00.png', cv2.IMREAD_GRAYSCALE)

    print img.shape[:2]

    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
