#----------------------------------------------------
# Sean Carda
# ECE 276A - Sensing and Estimation in Robotics
# Project 1
# 2/7/2022
#----------------------------------------------------
# This script grabs training data from the given folders and computes the optimal paramaters
# for classifying pixel values in real images. This file is identical to pixel trainer, but
# is renamed for cohesion.

#----------------------------------------------------
# IMPORTS
import os
import numpy as np
import cv2 as cv2
import matplotlib as mpl
from bin_detection.generate_rgb_data import read_pixels

#----------------------------------------------------
np.set_printoptions(suppress=True)

def train(folder_name, label_count):
    # Grab the training folders listed in the given directory.
    folders = np.flip(os.listdir(path=folder_name))
    
    # For each color in the training folder, read in the data.
    print('Loading the training data...')
    X_seperate = list()
    X = np.array([])
    for i in range(label_count):
        print('Current folder: ' + folders[i])
        X_seperate.append(read_pixels(folder_name + r'/' + folders[i]))

    # Concatenate the data into one matrix.
    X = np.array(X_seperate[0])
    for i in range(1, label_count):
        X = np.concatenate((X, np.array(X_seperate[i])))


    # Generate truth labels for the data we have been given.
    print('Loading the training labels...')
    Y = np.ones(len(X_seperate[0]))
    for i in range(1, label_count):
        Y = np.concatenate((Y, (i + 1) * np.ones(len(X_seperate[i]))))
    print('Done!')


    #------------------
    # THETA
    #------------------
    # First, we will find our optimal "maximum likelihood" estimates for the given labels.
    print('Calculating maximum likelihood theta...')
    theta = np.zeros(label_count)
    for i in range(label_count):
        # Determine the values of Y that match the current label index.
        indicator = (Y == (i + 1))

        # Sum the total matches and divide by the total number of samples.
        theta[i] = sum(indicator) / len(indicator)
    print('Done!')


    #------------------
    # MEAN
    #------------------
    print('Calculating optimal means...')
    # Next, we will calculate the mean given by pixels corresponding to each label.
    avg = np.zeros((3, label_count))
    for i in range(label_count):
        # Determine the vales of Y that match the current label index.
        indicator = (Y == (i + 1))
        ind_mat = indicator
        for j in range(2): 
            ind_mat = np.column_stack((ind_mat, indicator))

        # Consider data points relevant to the current label.
        valid_data = np.int64(ind_mat) * X

        # Sum this data.
        check_sum = np.sum(valid_data, axis=0) / sum(indicator)
        avg[:, i] = np.sum(valid_data, axis=0) / sum(indicator)
    print('Done!')


    #------------------
    # COVARIANCE
    #------------------
    # Finally, we will compute the covariance of the data.
    print('Calculating optimal covariance...')
    cov = list()
    for i in range(label_count):
        # Determine the values of Y that match the current label index.
        indicator = (Y == (i + 1))
        ind_mat = indicator
        for j in range(2): 
            ind_mat = np.column_stack((ind_mat, indicator))

        # Consider data points relevant to the current label.
        offset = np.transpose(np.int64(ind_mat) * (X - avg[:, i]))
        cov.append(np.dot(offset, np.transpose(offset)) / np.sum(indicator))
    print('Done!')

    # Return the trained parameters.
    np.savez(os.path.abspath('.') + '/bin_detection/bin_model_params.npz', t=theta, a=avg, c=cov)

