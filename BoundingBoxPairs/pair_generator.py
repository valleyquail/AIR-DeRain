import cv2 as cv
import os
import numpy as np
from bounding_box_generator import threshold_callback

original_path = "input/"
mask_path = "masks/"
blurred_path = "output/"
isolation_path = "isolated_drops/"

list_of_mask_batches = sorted(os.listdir(mask_path))

print(list_of_mask_batches)
if '.DS_Store' in list_of_mask_batches:
    list_of_mask_batches.remove('.DS_Store')

image_num = 0

for batch in list_of_mask_batches:
    current_original_folder = original_path + batch
    current_mask_folder = mask_path + batch
    current_blurred_folder = blurred_path + batch
    current_isolation_folder = isolation_path + batch
    if not os.path.exists(current_isolation_folder):
        os.mkdir(current_isolation_folder)
    # print("current_mask_folder:", current_mask_folder)
    # print("current_output_folder: ", current_output_folder)
    # print(os.listdir(current_mask_folder)[0:5])
    for image in os.listdir(current_mask_folder):
        original_image_path = current_original_folder + '/' + image
        mask_image_path = current_mask_folder + '/' + image
        blurred_image_path = current_blurred_folder + '/' + image
        isolated_drops_image_folder = current_isolation_folder + '/' + image
        if not os.path.exists(isolated_drops_image_folder):
            os.mkdir(isolated_drops_image_folder)
        # print(mask_image_path)
        bounding_boxes = threshold_callback(mask_image_path, 100)

        for i, box in enumerate(bounding_boxes):
            original_image = cv.imread(cv.samples.findFile(original_image_path))
            blurred_image = cv.imread(cv.samples.findFile(blurred_image_path))
            x, y, w, h = box

            original_region_of_interest = original_image[y:y + h, x:x + w]
            blurred_region_of_interest = blurred_image[y:y + h, x:x + w]

            cv.imwrite(isolated_drops_image_folder + "/original_area{}.png".format(i), original_region_of_interest)
            cv.imwrite(isolated_drops_image_folder + "/blurred_area{}.png".format(i), blurred_region_of_interest)
        print(image_num)
        image_num += 1
