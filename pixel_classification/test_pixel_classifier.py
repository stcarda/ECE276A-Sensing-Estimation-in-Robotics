'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


from __future__ import division

from generate_rgb_data import read_pixels
from pixel_classifier import PixelClassifier

if __name__ == '__main__':
  # test the classifier
  
  true_folder = 'data/validation/blue' 
  folder = r'C:\Users\Sean Carda\Desktop\ECE 276A - Sensing and Estimation in Robotics\Project 1\ECE276A_PR1\ECE276A_PR1\pixel_classification\data\training\blue'
  
  X = read_pixels(folder)
  myPixelClassifier = PixelClassifier()
  y = myPixelClassifier.classify(X)
  
  print('Precision: %f' % (sum(y==3)/y.shape[0]))

  
