import cv2

from main.feature.image_preprocessor import scale_to_fill

img_not_scale = cv2.imread('i.png', cv2.IMREAD_GRAYSCALE)
'''
height, width = img_not_scale.shape[:2]

# Get extreem values from the image
max_x = 0
min_x = width
max_y = 0
min_y = height

for x in range(0, width):
    for y in range(0, height):

        if img_not_scale[x][y] != 255:
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
# Cut out the part of image containing colored pixels
sub_image = img_not_scale[min_x:max_x, min_y:max_y]

# Scale the image
resize_image = cv2.resize(sub_image, (width, height), interpolation=cv2.INTER_CUBIC)
'''

resize_image = scale_to_fill(img_not_scale)
cv2.imwrite('hihi.png', resize_image)
cv2.imshow('c', img_not_scale)
cv2.imshow('b', resize_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
