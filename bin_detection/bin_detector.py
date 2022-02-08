'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import os
import numpy as np
import cv2
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
from bin_detection.bin_trainer import train

class BinDetector():
	def __init__(self):	
		# How many colors we wish to identify:
		self.color_count = 5

		# Train the model.
		# ** Uncomment to train the model.
		#folder_name = os.path.abspath('.') + '/bin_detection/data/training'
		#train(folder_name, self.color_count)

		self.segmented = 0
		
		# Grab the parameters saved by the training script.
		parameters = np.load(os.path.abspath('.') + '/bin_detection/bin_model_params.npz', allow_pickle=True)

		# Distinguish the parameters for calculations.
		self.theta = parameters['t']
		self.avg = parameters['a']
		self.cov = parameters['c']

		pass

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Function which utilizes the Gaussian model to compute the closest class corresponding 
		# to a given pixel.
		def compute_class(x):
			# Normalize the images.
			x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255

			# Grab the size of the current image.
			size = x.shape

			# Create an array of possible labels.
			Y = np.linspace(1, self.color_count, num=self.color_count)

			# Empty array of probabilities for a given label.
			p_y = np.zeros([size[0], size[1], self.color_count])
			for i in range(self.color_count):
				# Calculate p(y | theta) : 
				indicator = (Y == (i + 1))
				p_y_theta = 1
				for j in range(3):
					p_y_theta *= self.theta[j] ** np.int64(indicator[j])

				# Calculate p(x | y = k, w) :
				# Numerator of Gaussian.
				num = 1 / (np.sqrt(np.linalg.det(self.cov[i]) * (2 * np.pi)**3))

				# Convert the image to a N*M x 3 matrix so that it is more easily processible.
				x_reshape = np.transpose(np.reshape((x - self.avg[:, i]).flatten(), (size[0] * size[1], 3)))

				# Compute sigma * x.
				ex = np.dot(np.linalg.inv(self.cov[i]), x_reshape)

				# Compute the dot product of the original pixels of x with the vectors produced
				# after multiplying the pixels by the covariance.
				ex = np.einsum('ij,ji->i', np.transpose(x_reshape), ex)

				# Scale by -0.5 and reshape the matrix back into the size of the original.
				ex = -0.5 * np.reshape(ex, (size[0], size[1]))

				# Total probability for class i.
				p_y[:, :, i] = p_y_theta * num * np.exp(ex)

			# Return the class with the greatest probability for each entry in the new image.
			return np.int64(np.argmax(p_y, axis=2) + 1)

		# Compute the classification of the image.
		mask_img = compute_class(img)

		# Save the segmented image for bounding boxes.
		self.segmented = mask_img
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		# Return statement for an eroded segmentation.
		#return [mask_img, cv2.erode(np.uint8(np.where(mask_img == 5, 255, 0)), np.ones((11, 11)))]
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Grab the size of the image.
		size = img.shape

		# Extract the regions which we have labeled as blue.
		blue = np.uint8(np.where(img == 5, 255, 0))

		# Label the segmented image and compute some properties of this image.
		blue_labeled = label(blue)
		#blue_labeled = label(cv2.erode(blue, np.ones((11, 11))))
		blue_props = regionprops(blue_labeled)

		# For every object we have calculated the properties for, find the object with the
		# greatest area.
		areas = np.zeros(len(blue_props))
		for i in range(len(blue_props)):
			areas[i] = blue_props[i].area

		# Find the label of the object of the highest area.
		max_labels = np.argwhere((areas > 1000) & (areas < 650000))

		# Find the areas of regions with perimeters that are similar to bins.
		boxes = []
		for region in max_labels:
			# Find where the blue labeled image is nonzero.
			im_label = np.where(blue_labeled == region + 1, 1, 0)
			im_nonzero = np.argwhere(im_label > 0)

			# Grab the necessary corners of the bounded region.
			r_max = np.max(im_nonzero[:, 0])
			r_min = np.min(im_nonzero[:, 0])
			c_max = np.max(im_nonzero[:, 1])
			c_min = np.min(im_nonzero[:, 1])

			# Filter the bounded regions by ratio.
			ratio = (r_max - r_min) / (c_max - c_min)
			if (3 > ratio > 1):
				boxes.append([c_min, r_min, c_max, r_max])
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		
		return boxes


