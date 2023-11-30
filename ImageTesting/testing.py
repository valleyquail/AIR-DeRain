import fastradialsymmetry as frst
import cv2 as cv


img = cv.imread(cv.samples.findFile("../WholeImageData/train/data/18_rain.png"))

src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

output = frst.frst(src_gray, 30, 1, .9, 2/200, 'DARK')

cv.imshow("Final", output)
cv.waitKey(0)
