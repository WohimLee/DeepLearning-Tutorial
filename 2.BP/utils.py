import os
import struct
import hashlib
import numpy as np

from pathlib import Path


def load_labels(file):
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_of_images = struct.unpack(">ii", data[:8])
    if magic_number != 2049:   # 0x00000801
        print(f"magic number mismatch {magic_number} != 2049")
        return None
    
    labels = np.array(list(data[8:]))
    return labels

def load_images(file):
    with open(file, "rb") as f:
        data = f.read()

    magic_number, num_of_images, rows, columns = struct.unpack(">iiii", data[:16])
    if magic_number != 2051:   # 0x00000803
        print(f"magic number mismatch {magic_number} != 2051")
        return None
    
    images = np.asarray(list(data[16:]), dtype=np.uint8).reshape(num_of_images, -1)
    return images


def one_hot(labels, classes, label_smoothing=0):
    n = len(labels)
    eoff = label_smoothing / classes
    output = np.ones((n, classes), dtype=np.float32) * eoff
    for row, label in enumerate(labels):
        output[row, label] = 1 - label_smoothing + eoff
    return output


def norm_images(images):
    variance = np.var(images)
    mean     = np.mean(images)
    return (images - mean) / variance


def get_md5(data):
    return hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()



def mkparent(path):
    parents = Path(path).parent
    os.makedirs(parents, exist_ok=True)


if __name__ == "__main__":
    train_images = "/datav/Dataset/MNIST/train-images-idx3-ubyte"
    train_labels = "/datav/Dataset/MNIST/train-labels-idx1-ubyte"
    test_images  = "/datav/Dataset/MNIST/t10k-images-idx3-ubyte"
    test_labels  = "/datav/Dataset/MNIST/t10k-labels-idx1-ubyte"

    # train_images = load_images(train_images)
    # train_labels = load_labels(train_labels)
    test_images  = load_images(test_images)
    test_labels  = load_labels(test_labels)
    print()






