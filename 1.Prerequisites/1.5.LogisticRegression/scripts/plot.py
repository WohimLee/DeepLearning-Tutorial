
import numpy as np

import matplotlib.pyplot as plt

X = np.random.randint(0, 10, 20) + np.random.rand(20,)
X = np.sort(X)

negative = X[X<5]
positive = X[X>=5]
Y = [0]*len(negative)+[1]*len(positive)

plt.title('')
plt.xlabel('Tumor Size')
plt.plot(negative, Y[:len(negative)], "gx")
plt.plot(positive, Y[len(negative):], "ro")

plt.show()


