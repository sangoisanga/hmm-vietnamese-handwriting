import cv2

img_not_scale = cv2.imread('not_scale.png',cv2.IMREAD_GRAYSCALE)

height, width = img_not_scale.shape[:2]

# Get extreem values from the image
max_x = 0
min_x = height
max_y = 0
min_y = width

for x in range(0, height):
    for y in range(0, width):

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
sub_image = img_not_scale[min_y:max_y, min_x:max_x]

# Scale the image
resize_image = cv2.resize(sub_image, (width, height), interpolation=cv2.INTER_CUBIC)

cv2.imshow('c', img_not_scale)
cv2.imshow('a', sub_image)
cv2.imshow('b', resize_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
