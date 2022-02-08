#----------------------------------------------------
# Sean Carda
# ECE 276A - Sensing and Estimation in Robotics
# Project 1
# 2/7/2022
#----------------------------------------------------
# This script is for generating the training dataset for recycling bin classification.
# Here, we use RoiPoly to grab certain regions of the provided training images and 
# take the average pixel value of the region we grab. We take 10 samples per
# training image.

#----------------------------------------------------
# IMPORTS
from operator import contains
import os
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
from bin_detection.roipoly import RoiPoly
#----------------------------------------------------
np.set_printoptions(suppress=True)

# Here, we will set the color we wish to identify. Then, we will store the pixels in the
# corresponding folder in the training data folder.
color = r'gray'
read_folder = os.path.abspath('.') + '/bin_detection/data/training'
save_folder = read_folder + '/' + color

# For every image that already exists in the training data folder, we will read it in and
# mark regions of interest for the corresponding color. Most of the code was borrowed from 
# # the test_roipoly.py file. 
image_num = 0
for filename in os.listdir(read_folder):
    # If the current item in the folder is not another folder.
    if contains(filename, '.jpg'):
        # Initialize the number of samples to be taken from the image.
        samples = 0

        # Read in the image.
        im = cv2.cvtColor(cv2.imread(read_folder + r'/' +  filename), cv2.COLOR_BGR2RGB)
        plt.figure(1)
        plt.imshow(im)
        plt.show()

        # If we wish to skip this image due to a lack of desired color samples, then we
        # will move on to the next image in the folder.
        val = input('Skip this image? : ')
        if val == 'y':
            image_num += 1
            continue
        
        file_num = 10 * image_num
        # If we do not wish to skip the image, then we will continue with sample aquisition.
        while(samples < 10):
            # Read in the image from BGR to RGB.
            print(read_folder + r'/' +  filename)
            im = cv2.cvtColor(cv2.imread(read_folder + r'/' +  filename), cv2.COLOR_BGR2RGB)
            
            # Show the image.
            fig, ax = plt.subplots()
            ax.imshow(im)

            # Create a new roipoly object for the image.
            im_roi = RoiPoly(fig=fig, ax=ax, color='r')

            # Create a mask for the image and show it.
            mask = im_roi.get_mask(im)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('%d pixels selected\n' % im[mask,:].shape[0])
            ax1.imshow(im)
            ax1.add_line(plt.Line2D(im_roi.x + [im_roi.x[0]], im_roi.y + [im_roi.y[0]], color=im_roi.color))
            ax2.imshow(mask)
            plt.show()

            # From the mask, we compute the average of the region of the image where the mask
            # is nonzero.
            avg = np.zeros([1, 1, 3])
            for i in range(3):
                region = mask * im[:, :, i]
                avg[0, 0, i] = np.mean(region[np.nonzero(region)])

            # Write the average pixel as an image to the folder corresponding to the current
            # color. 
            cv2.imwrite(os.path.join(save_folder, color + str(file_num) + '.jpg'), cv2.cvtColor(np.uint8(avg), cv2.COLOR_RGB2BGR))

            # Increment sample and file number.
            samples += 1
            file_num += 1

        # Increment the image number.
        image_num += 1

    # If we've run into a folder, stop the collection process.
    else:
        break












