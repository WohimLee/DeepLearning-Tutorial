
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gd_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


x = np.linspace(-6, 6, 200)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))


ax.spines[['left', 'top']].set_position('zero')
ax.spines[['right', 'bottom']].set_visible(False)
ax.set(title='sigmoid')
y = sigmoid(x)
ax.plot(x, y)

y = gd_sigmoid(x)
ax.plot(x, y)

plt.show()