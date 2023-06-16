
import struct
import numpy as np


def mnist_labels(path):
    with open(path, "rb") as f:
        data = f.read()
    _, num_item = struct.unpack_from(">II", data, 0)
    labels = struct.unpack_from("B"*num_item, data, 8)
    return np.array(labels)
    
def mnist_images(path):
    with open(path, "rb") as f:
        data = f.read()
    _, num_item, rows, cols = struct.unpack_from(">iiii", data, 0)
    images = struct.unpack_from("B"*num_item*rows*cols, data, 16)
    return np.array(images).reshape(num_item, -1)


def one_hot(labels, classes=10, label_smoothing=0):

    n = len(labels)
    output = np.zeros((n, classes), dtype=np.float32)
    rows = np.arange(n)
    cols = labels
    output[rows, cols] = 1
    return output

def one_hot_smoothing(labels, classes=10, label_smoothing=0):
    '''
    Parameters:
        labels : ndarray, 标签数据
        classes: int, 类别
        label_smoothing: float, 标签平滑, 取值: 0~1
    '''
    n = len(labels)
    eoff = label_smoothing / classes
    output = np.ones((n, classes), dtype=np.float32) * eoff
    rows = np.arange(n)
    cols = labels
    output[rows, cols] = 1 - label_smoothing + eoff
    return output
