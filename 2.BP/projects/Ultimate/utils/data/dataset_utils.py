
import struct
import numpy as np


def load_labels(file):
    with open(file, "rb") as f:
        data = f.read()
    
    magic_number, num_samples = struct.unpack(">ii", data[:8])
    if magic_number != 2049:   # 0x00000801
        print(f"magic number mismatch {magic_number} != 2049")
        return None
    
    labels = np.frombuffer(data[8:], dtype=np.uint8)
    return labels

def load_images(file):
    with open(file, "rb") as f:
        data = f.read()

    magic_number, num_samples, image_width, image_height = struct.unpack(">iiii", data[:16])
    if magic_number != 2051:   # 0x00000803
        print(f"magic number mismatch {magic_number} != 2051")
        return None
    
    image_data = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_samples, -1)
    # return image_data.reshape(-1, image_height, image_width)
    return image_data


def one_hot(labels, classes=10, label_smoothing=0):
    n = len(labels)
    eoff = label_smoothing / classes
    output = np.ones((n, classes), dtype=np.float32) * eoff
    output[np.arange(n), labels] = 1 - label_smoothing + eoff
    return output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def estimate_val(predict, gt_labels, classes, loss_func):
    plabel = predict.argmax(1)
    positive = plabel == gt_labels
    total_images = predict.shape[0]
    accuracy = sum(positive) / total_images
    return accuracy, loss_func(predict, one_hot(gt_labels, classes))

def lr_schedule_cosine(lr_min, lr_max, per_epochs):
    def compute(epoch):
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / per_epochs * np.pi))
    return compute