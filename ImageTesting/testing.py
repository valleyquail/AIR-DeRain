import cv2 as cv

import numpy as np

img = cv.imread(cv.samples.findFile("../WholeImageData/train/data/30_rain.png"))
# img = cv.imread(cv.samples.findFile("../BoundingBoxPairs/output/30_blur.png"))
num_superpixels = 1000
prior = 2
num_levels = 3
num_iterations = 4
num_histogram_bins = 10

converted_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
height, width, channels = converted_img.shape

output = seeds = cv.ximgproc.createSuperpixelSEEDS(width, height, channels,
                                                   num_superpixels, num_levels, prior, num_histogram_bins)

seeds.iterate(converted_img, num_iterations)
color_img = np.zeros((height, width, 3), np.uint8)
color_img[:] = (0, 0, 255)
number_of_super = seeds.getNumberOfSuperpixels()

labels = seeds.getLabels()
mask = seeds.getLabelContourMask(False)
mask_inv = cv.bitwise_not(mask)
result_bg = cv.bitwise_and(img, img, mask=mask_inv)
result_fg = cv.bitwise_and(color_img, color_img, mask=mask)
result = cv.add(result_bg, result_fg)
cv.imshow("super pixel", result)

cv.waitKey(0)
