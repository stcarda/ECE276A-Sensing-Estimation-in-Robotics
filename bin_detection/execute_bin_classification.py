#----------------------------------------------------
# Sean Carda
# ECE 276A - Sensing and Estimation in Robotics
# Project 1
# 2/7/2022
#----------------------------------------------------
# This script aims to execute various tests on images to generate examples in the final report.

#----------------------------------------------------
# IMPORTS
import numpy as np
import cv2
import os
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
from bin_detection.bin_detector import BinDetector
from skimage import morphology
#----------------------------------------------------

# Project file from which to read in images. Specific to this laptop.
project_file_name = r'C:/Users/Sean Carda/Desktop/ECE 276A - Sensing and Estimation in Robotics/Project 1/ECE276A_PR1/ECE276A_PR1'

# Read in a specific image from the bin validation folder and convert it from BGR to RGB.
for file_num in range(61, 71):
    im = cv2.imread(project_file_name + '/bin_detection/data/validation/00' + str(file_num) + '.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Instantiate a bin detector.
    detect = BinDetector()

    # Setment and display the image.
    segmented = detect.segment_image(im)
    plt.figure()
    plt.imshow(segmented)
    plt.colorbar()
    plt.savefig(os.path.abspath('.') + '/Excecute Tests/segment' + str(file_num) + '.png')
    plt.close()

    # Calculate the valid boxes on the regions where the classifier believes there are bins.
    boxes = detect.get_bounding_boxes(segmented)

    # Draw boxes around valid regions. Code borrowed from the bin testing code provided to us.
    boxed_img = im
    for i in range(len(boxes)):
        # Grab the current box.
        current_box = boxes[i]
        
        # Grab the relevant corners.
        corner_one = (current_box[0], current_box[1])
        corner_two = (current_box[2], current_box[3])
        boxed_img = cv2.rectangle(im, corner_one, corner_two, (255, 0, 0), 3)

    # Display the original image with self-generated boxes.
    plt.figure()
    plt.imshow(boxed_img)
    plt.colorbar()
    plt.savefig(os.path.abspath('.') + '/Excecute Tests/boxed' + str(file_num) + '.png')
    plt.close()
    
    print('file: ' + str(file_num))
    print(boxes)

