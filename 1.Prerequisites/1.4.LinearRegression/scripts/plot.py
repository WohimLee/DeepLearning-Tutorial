
import random
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10, dtype=np.float32) + 2009
# y = np.arange(10, dtype=np.float32)+np.random.randn(10)
y = np.array([[1.8, 2.1, 2.3, 2.3, 2.85, 3.0, 3.3, 4.9, 5.45, 5.0]], dtype=np.float32).T


tx = np.array([2009, 2018])
ty = np.array([1, 5.5])

plt.xlabel('Year')
plt.ylabel('Price')
plt.title("Hose Price (2009-2018)")
plt.grid()

plt.plot(x, y, ".")
plt.savefig('../imgs/houseprice-raw.png')

plt.plot(tx, ty, 'g-')
plt.savefig('../imgs/houseprice-init.png')

# plt.show()