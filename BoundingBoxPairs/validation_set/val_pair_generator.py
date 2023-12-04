import cv2 as cv
import os
import re
from val_bounding_box_generator import threshold_callback

original_path = "input/"
mask_path = "masks/"
blurred_path = "output/"
isolation_path = "isolated_drops/"

image_num = 0

if not os.path.exists(isolation_path):
    os.mkdir(isolation_path)
# print("current_mask_folder:", current_mask_folder)
# print("current_output_folder: ", current_output_folder)
# print(os.listdir(current_mask_folder)[0:5])
idx = 0
for image in os.listdir(mask_path):
    original_image_path = original_path + image.replace('mask', 'clean')
    mask_image_path = mask_path + image
    blurred_image_path = blurred_path + image.replace('mask', 'blur')

    if not os.path.exists(isolation_path):
        os.mkdir(isolation_path)
    if not os.path.exists(isolation_path + 'clean'):
        os.mkdir(isolation_path + 'clean')
    if not os.path.exists(isolation_path + 'blurred'):
        os.mkdir(isolation_path + 'blurred')
    # print(mask_image_path)
    bounding_boxes = threshold_callback(mask_image_path, 100)

    for i, box in enumerate(bounding_boxes):
        original_image = cv.imread(cv.samples.findFile(original_image_path))
        blurred_image = cv.imread(cv.samples.findFile(blurred_image_path))
        x, y, w, h = box

        original_region_of_interest = original_image[y:y + h, x:x + w]
        blurred_region_of_interest = blurred_image[y:y + h, x:x + w]
        idx += 1
        cv.imwrite(isolation_path + 'clean' + "/{}_clean.png".format(idx), original_region_of_interest)
        cv.imwrite(isolation_path + 'blurred' + "/{}_blur.png".format(idx), blurred_region_of_interest)

    print(image_num)
    image_num += 1
