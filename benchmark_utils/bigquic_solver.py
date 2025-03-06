import numpy as np
from py_bigquic.py_bigquic import bigquic

# A test data array with 10 samples, 20 variables
data_array = np.random.random((10, 20))
alpha = 0.5  # The sparsity hyper parameter
prec = bigquic(data_array, alpha)  # Returns the precision matrix
