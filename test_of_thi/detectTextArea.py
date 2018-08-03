import glob
import os
from os.path import join
import cv2 as cv
import numpy as np


def boundRects(r1, r2):
    result = r1[:]
    w = r2[1] + r2[2] - result[1]
    h = r2[0] + r2[3] - result[0]
    if w > result[2]:
        result[2] = w
    if h > result[3]:
        result[3] = h
    return result


def position(r1, r2):
    """rectagle is an array with 4 value: x, y, w, h"""
    x, y = (r2[0] - r1[0], r2[1] - r1[1])
    if x >= 0 and y >= 0:
        if r1[2] >= y and r1[3] >= x:
            return boundRects(r1, r2)


def main():
    base_dir = os.path.abspath('.')
    files = glob.glob1(base_dir, '*.jpg')
    #files = ["A.jpg"]
    output_dir = join(base_dir, 'out')
    for f in files:
        mser = cv.MSER_create()
        im = cv.imread(join(base_dir, f))

        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        regions, _ = mser.detectRegions(gray)
        hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        out = im.copy()

        for c in hulls:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)

        mask = np.zeros((im.shape[0], im.shape[1], 1), dtype=np.uint8)
        for contour in hulls:
            cv.drawContours(mask, [contour], -1, (255, 255, 255), -1)

        idx = 0
        for c in hulls:
            x, y, w, h = cv.boundingRect(c)
            idx += 1
            new_img = im[y:y + h, x:x + w]
            name_file = f[0] + str(idx) + '.png'
            cv.imwrite(join(output_dir, name_file), new_img)


if __name__ == '__main__':
    main()
