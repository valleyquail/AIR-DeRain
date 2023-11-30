import cv2 as cv
import sys
import numpy as np

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 4
kernel_size = 5


def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3, 3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = img * (mask[:, :, None].astype(img.dtype))
    cv.imshow(window_name, dst)


img = cv.imread(cv.samples.findFile("../WholeImageData/train/data/12_rain.png"))

if img is None:
    sys.exit("Could not find the image")

cv.namedWindow("Input", cv.WINDOW_AUTOSIZE)
# cv.namedWindow("Output", cv.WINDOW_AUTOSIZE)

cv.imshow("Input", img)
# cv.waitKey(0)
# sys.exit("sss")

alpha = 2.0
beta = 1
new_img = cv.convertScaleAbs(img, alpha, beta)

kernel = np.array([[-3, 0, 0, 0, 3],
                   [0, -2, 0, 2, 0],
                   [0, 0, 1, 0, 0],
                   [0, -2, 0, 2, 0],
                   [-3, 0, 0, 0, 3]], np.float32)  # kernel should be floating point type

output = cv.filter2D(new_img, -1, kernel)

kernel2 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]], np.float32)

output = cv.filter2D(output, -1, kernel2)

src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name, 0, max_lowThreshold, CannyThreshold)


CannyThreshold(5)
# cv.imshow("Output", output)

cv.waitKey(0)
cv.destroyAllWindows()
