import cv2 as cv
import sys
import numpy as np
import random as rng

rng.seed(12345)

dilation_size = 10
dilation_shape = cv.MORPH_ELLIPSE
dilation_kernel = cv.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                           (dilation_size, dilation_size))


def threshold_callback(mask_path, val):
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

    # drawing = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    #
    # for i in range(len(contours)):
    #     color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    #     cv.drawContours(drawing, contours_poly, i, color)
    #     cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])),
    #                  (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
    #     cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    #
    # cv.imshow('Contours', drawing)

    return boundRect


if __name__ == "__main__":
    threshold_callback(50)

    cv.waitKey(0)
    sys.exit()
