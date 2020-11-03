'''
Metrics used to estimate ...

'x_pred' and 'x_ref' are 1-dimensional arrays and have same dimension.
'''

import numpy as np

def mse(x_pred, x_ref, nbVar = 1):

    '''
    Mean Squared Error
    '''

    N = len(x_pred)
    return (1 / (N * nbVar)) * np.sum((x_pred - x_ref)**2)

def rmse(x_pred, x_ref, nbVar = 1):

    '''
    Root Mean Squared Error
    Reference metric used for evaluation of MEX Power Challenge.
    '''

    return np.sqrt(mse(x_pred, x_ref, nbVar))

def mae(x_pred, x_ref):

    '''
    Mean Absolute Error
    '''

    N = len(x_pred)
    return (1 / N) * np.sum(np.abs(x_pred - x_ref))

def scr(x_pred, x_ref):

    '''
    Sum of Squared Errors
    '''

    return np.sum((x_pred - x_ref)**2)
