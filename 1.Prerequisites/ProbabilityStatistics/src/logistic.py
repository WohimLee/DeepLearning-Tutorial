

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-7, 7, 300)

def gaussianPDF(x, mean=0, std=1):
    term1 = 1 / (std*np.sqrt(2*np.pi))
    term2 = np.exp(-1/(2*std**2)*(x - mean)**2)
    return term1*term2


def logisticPDF(x, mu=0, gamma=1):
    term1 = np.exp(-(x-mu)/gamma)
    term2 = gamma*(1+term1)**2
    return term1 / term2


def logisticCDF(x, mu=0, gamma=1):
    out = 1 / (1+np.exp(-(x-mu)/gamma))
    return out

fig, axes = plt.subplots(1, 2, figsize=(10, 5))


ax1 = axes[0]
ax1.set(title='Logistic PDF')
ax1.spines[['left', 'top']].set_position('zero')
ax1.spines[['right', 'bottom']].set_visible(False)

y1 = logisticPDF(x)
ax1.plot(x, y1)

ax2 = axes[1]
ax2.set(title='Logistic CDF (sigmoid)')
ax2.spines[['left', 'top']].set_position('zero')
ax2.spines[['right', 'bottom']].set_visible(False)


y2 = logisticCDF(x)
ax2.plot(x, y2)

fig.suptitle('Logistic Distribution')
fig.tight_layout()


plt.show()
fig.savefig('../imgs/logistic.png')



