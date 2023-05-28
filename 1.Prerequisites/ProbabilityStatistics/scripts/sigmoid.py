
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gd_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))