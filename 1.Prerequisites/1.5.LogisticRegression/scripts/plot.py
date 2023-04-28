
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
X = np.random.randint(0, 10, 20) + np.random.rand(20,)
X = np.sort(X)

negative = X[X<5]
positive = X[X>=5]
Y = [0]*len(negative)+[1]*len(positive)

plt.title('')
plt.xlabel('Tumor Size')
plt.plot(negative, Y[:len(negative)], "gx")
plt.plot(positive, Y[len(negative):], "ro")
plt.grid()
plt.savefig('../imgs/tumor-raw.png')


plt.clf()
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

sigmoid_x = np.linspace(-10, 10, 100)
sigmoid_y = sigmoid(sigmoid_x)
plt.title('sigmoid')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(sigmoid_x, sigmoid_y)
plt.grid()
plt.savefig('../imgs/sigmoid.png')


plt.clf()
Y = sigmoid(X)
plt.title('')
plt.xlabel('Tumor Size')
plt.plot(X, Y, 'o')
plt.grid()
plt.savefig('../imgs/tumor-sigmoid.png')

# plt.show()


