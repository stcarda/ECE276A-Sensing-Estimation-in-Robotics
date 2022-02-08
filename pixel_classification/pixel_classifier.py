'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
import os

from pixel_trainer import train

class PixelClassifier():
  def __init__(self):
    # How many colors we wish to identify:
    self.color_count = 3
    
    # Determine the folder from which to read in data.
    # ** Uncomment for training.
    #folder_name = os.path.abspath('.') + '/pixel_classification/data/training'
    #train(folder_name, self.color_count)
    
    # Grab the parameters saved by the training script.
    parameters = np.load(os.path.abspath('.') + '/pixel_classification/pixel_model_params.npz', allow_pickle=True)

		# Distinguish the parameters for calculations.
    self.theta = parameters['t']
    self.avg = parameters['a']
    self.cov = parameters['c']

    pass
	
  def classify(self,X):

    ################################################################
    # YOUR CODE AFTER THIS LINE
    
    # Function which utilizes the Gaussian model to compute the closest class corresponding 
    # to a given pixel.
    def compute_class(x):
      Y = np.linspace(1, self.color_count, num=self.color_count)
      p_y = np.zeros(self.color_count)
      for i in range(self.color_count):
        # Calculate p(y | theta) : 
        indicator = (Y == (i + 1))
        p_y_theta = 1
        for j in range(3):
          p_y_theta *= self.theta[j] ** np.int64(indicator[j])

        # Calculate p(x | y = k, w) :
        # Numerator of Gaussian.
        num = 1 / (np.sqrt(np.linalg.det(self.cov[i]) * (2 * np.pi)**3))
        
        # Exponent of Gaussian.
        ex = -0.5 * np.dot(np.transpose(x - self.avg[:, i]), np.dot(np.linalg.inv(self.cov[i]), (x - self.avg[:, i])))

        # Total probability for class i.
        p_y[i] = p_y_theta * num * np.exp(ex)

      # Return the class with the greatest probability.
      return int(np.argmax(p_y) + 1)

    # For every given test input, compute the class that is closest to the true color.
    y = np.int64(np.zeros(len(X)))
    for i in range(0, len(X)):
      y[i] = compute_class(X[i, :])
    
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y
