import cv2
import numpy as np

from main.feature.image_preprocessor import scale_to_fill


img_not_scale = cv2.imread('not_scale.png',cv2.IMREAD_GRAYSCALE)


test = scale_to_fill(img_not_scale)



cv2.imshow('b',img_not_scale)
cv2.imshow('a',test)
cv2.waitKey(0)
cv2.destroyAllWindows()
