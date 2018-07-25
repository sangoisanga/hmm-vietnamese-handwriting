import cv2
import numpy as np


img = np.zeros((100,100))

img[80][20] = 255


cv2.imshow('a',img)
cv2.waitKey(0)
cv2.destroyAllWindows()