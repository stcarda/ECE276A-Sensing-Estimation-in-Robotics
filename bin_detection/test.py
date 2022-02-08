import numpy as np
import os

params = np.load(os.path.abspath('.') + '/bin_detection/bin_model_params.npz')

theta = params['t']
avg = params['a']
cov = params['c']

print(cov)