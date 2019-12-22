import numpy as np
# mean squared error
def MSE(actual,predicted):
    return np.sqrt(np.mean((predicted - actual)**2 ))
