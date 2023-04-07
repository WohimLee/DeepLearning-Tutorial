

import numpy as np
import matplotlib.pyplot as plt

predict = np.array([1.5, 2.1, 3.8, 4.2, 5.0])
target  = np.array([1.2, 2.0, 3.5, 4.0, 5.5])


def MSELoss(predict, target):
    loss = np.mean((predict - target) ** 2)
    return loss

# plot the data
plt.scatter(target, predict, color='blue', label='Data')

# plot the line of best fit
m, b = np.polyfit(target, predict, 1)
plt.plot(target, m*target + b, color='red', label='Line of best fit')

plt.xlabel('Target Values')
plt.ylabel('Predicted Values')
plt.title('MSE Loss')
plt.legend()
plt.show()

mse_loss = MSELoss(predict, target)

print(mse_loss)
