import cv2 as cv
import sys


src = cv.imread(cv.samples.findFile("../SyntheticData/SyntheticMasks/1o_0000.png"))

cv.imshow("Original", src)
cv.waitKey(0)
sys.exit()

