import time
import random
import numpy as np
import matplotlib.pyplot as plt
random.seed(1888)



def draw_line(k, b):
    x1 = 0
    y1 = k*x1 + b
    x2 = 10
    y2 = k*x2 + b
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot([x1, x2], [y1, y2], 'g-')
    plt.show()

def train(epoch=1000, lr=1e-2):

    k = random.random()
    b = 0
    for i in range(epoch):
        predict = x*k + b
        loss = 0.5*np.sum((y - predict)**2)
        if i % 200 == 0:
            print(f"Loss: {loss}")
            plt.plot(x, y, ".")
            draw_line(k, b)
            time.sleep(1)
        delta_b = np.mean(predict - y)
        delta_k = np.mean((predict - y) * x)
        k = k - lr * delta_k
        b = b - lr * delta_b
    return k, b

if __name__ == '__main__':
    x = np.arange(10, dtype=np.float32)
    y = np.arange(10, dtype=np.float32)+np.random.randn(10)
    train()
    
    
