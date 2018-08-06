import time

import cv2
import numpy as np

img = cv2.imread('segment0.png', cv2.IMREAD_GRAYSCALE)

print img.shape

height, width = img.shape[:2]
max_height_left = None
max_height_right = None

max_height_left_pos = None
max_height_right_pos = None

for i in range(width):
    if max_height_left is None:
        for j in range(height):
            if img[j][i] != 255:
                max_height_left = j
                max_height_left_pos = i
                break
    if max_height_right is None:
        temp_i = -(i + 1)
        for j in range(height):
            if img[j][temp_i] != 255:
                max_height_right = j
                max_height_right_pos = temp_i
                break

    if max_height_right is not None and max_height_left is not None:
        break

print max_height_left
print max_height_left_pos
print max_height_right
print max_height_right_pos




cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
