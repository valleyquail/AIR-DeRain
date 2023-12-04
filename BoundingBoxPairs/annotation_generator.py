import os

import cv2 as cv
import sys
import numpy as np
import random as rng
import csv

rng.seed(12345)

dilation_size = 10
dilation_shape = cv.MORPH_ELLIPSE
dilation_kernel = cv.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                           (dilation_size, dilation_size))


def threshold_callback(mask_path, val):
    # print(mask_path)
    mask = cv.imread(cv.samples.findFile(mask_path))
    # output = cv.imread(cv.samples.findFile(output_path))
    threshold = val
    mask_dilation = cv.dilate(mask, dilation_kernel)
    canny_output = cv.Canny(mask_dilation, threshold, threshold * 2)
    # cv.imshow("Canny", canny_output)
    # mask = np.array()
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 10, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    return boundRect


if __name__ == "__main__":

    header = ['Image']
    for i in range(50):
        header.append('x_top{}'.format(i))
        header.append('y_top{}'.format(i))
        header.append('x_bottom{}'.format(i))
        header.append('y_bottom{}'.format(i))
    print(header)
    file = "Annotations/annotations.csv"
    with open(file, 'w') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(header)
        images = os.listdir("masks/")
        images = sorted(images)
        for image in images:
            path = "masks/" + image
            # print(path)
            rectangle = threshold_callback(path, 50)
            image = image.replace('mask', 'blur')
            coords = []
            for s in rectangle:
                for x in s:
                    coords.append(x)
            writer.writerow([image, *coords])


