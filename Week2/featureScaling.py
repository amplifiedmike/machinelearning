import numpy as np

def featureScaling(x,fs_type):
    x_mu = np.mean(x)
    x_max = np.max(x)
    x_min = np.min(x)

    x1_meanNormalization = (x-x_mu)/(x_max-x_min)
    return x1_meanNormalization
