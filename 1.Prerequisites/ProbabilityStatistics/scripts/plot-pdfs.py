
import numpy as np
import matplotlib.pyplot as plt



x = np.linspace(-4*np.pi, 4*np.pi, 200)

y1 = x*np.cos(x)
y2 = np.exp(x) - np.sin(x)
y3 = x - np.cos(x)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax1 = axes[0]
ax1.spines[['left', 'top']].set_position('zero')
ax1.spines[['right', 'bottom']].set_visible(False)
ax1.grid()
ax1.plot(x, y1)

ax2 = axes[1]
ax2.spines[['left', 'top']].set_position('zero')
ax2.spines[['right', 'bottom']].set_visible(False)
ax2.grid()
ax2.set_xlim(-15, 5)
ax2.set_ylim(-4, 10)
ax2.plot(x, y2)

ax3 = axes[2]
ax3.spines[['left', 'top']].set_position('zero')
ax3.spines[['right', 'bottom']].set_visible(False)
ax3.grid()
ax3.plot(x, y3)


plt.show()
fig.savefig('../imgs/pdfs.png')