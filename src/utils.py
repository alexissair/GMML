import numpy as np

def l2(x, y) :
    return (x - y)**2

def l1(x, y) :
    return np.abs(x - y)