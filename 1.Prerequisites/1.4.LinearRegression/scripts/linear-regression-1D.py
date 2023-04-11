import time
import random
import numpy as np
import matplotlib.pyplot as plt
# random.seed(1888)

def draw_line(k, b):
    x1 = 0
    y1 = k*x1 + b
    x2 = 10
    y2 = k*x2 + b
    
    plt.plot([x1, x2], [y1, y2], 'g-')


def train(epoch=1000, lr=1e-2):

    k = random.random()
    b = 0
    plt.figure()
    for i in range(epoch):
        predict = x*k + b
        loss = 0.5*np.sum((y - predict)**2)
        delta_b = np.mean(predict - y)
        delta_k = np.mean((predict - y) * x)
        k = k - lr * delta_k
        b = b - lr * delta_b
        
        tx = np.array([0, 1])
        ty = k * tx + b
        if i % 100 == 0:
            plt.clf()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title("Iter %d : loss=%f, k=%f, b=%f"%(i, loss, k, b))
            plt.plot(x, y, ".")
            plt.plot(tx, ty, 'g-')
            plt.axis([0, 1, 0, 1])
            plt.pause(0.5)
        
    return k, b

if __name__ == '__main__':
    x = np.arange(10, dtype=np.float32)
    y = np.arange(10, dtype=np.float32)+np.random.randn(10)
    # y = np.array([[1.8, 2.1, 2.3, 2.3, 2.85, 3.0, 3.3, 4.9, 5.45, 5.0]], dtype=np.float32).T
    
    x = x / len(x)
    y = y / len(y)
    # plt.scatter(x, y)
    # plt.title("House Price")
    # plt.grid()
    # plt.savefig("./houseprice.png")
    
    k, b = train()
    #估算2019年的房价多少
    #归一化
    x_2019 = (2019-2009) / 10.0
    v_2019 = x_2019 * k + b

    #结果反归一化
    v_2019 = v_2019 * 10
    print("模型的参数是：k=%f, b=%f, 预估的2019房价为：%f 万元" % (k, b, v_2019))
    
    
