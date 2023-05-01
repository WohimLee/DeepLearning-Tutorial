
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import AxesZero


fig = plt.figure()
ax = fig.add_subplot(axes_class=AxesZero)

for direction in ["xzero", "yzero"]:
    # adds arrows at the ends of each axis
    ax.axis[direction].set_axisline_style("-|>")

    # adds X and Y-axis from the origin
    ax.axis[direction].set_visible(True)

for direction in ["left", "right", "bottom", "top"]:
    # hides borders
    ax.axis[direction].set_visible(False)

xlim = (-5, 5)
ylim = (-7, 7)

k, b = 1, 0.2
x = np.linspace(-5, 5, 100)
noise = np.random.randn(100)
y = k*x + b + noise


plt.xlim(xlim)
plt.ylim(ylim)

plt.grid()
# plt.savefig('imgs/linear-1.png')
plt.show()




