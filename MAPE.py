import numpy as np
# mean absolute percentage error
def MAPE(actual,predicted):
    return (100.0/len(actual))*np.sum(np.abs(predicted-actual)/actual)
