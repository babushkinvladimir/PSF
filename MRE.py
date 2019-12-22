import numpy as np
# mean relative error
def MRE(actual,predicted):
    return (100.0 / len(actual)) * np.sum(np.abs(predicted - actual) / np.mean(actual))
