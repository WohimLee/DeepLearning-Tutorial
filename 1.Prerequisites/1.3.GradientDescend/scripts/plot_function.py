

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

x = np.linspace(-5, 5, 100)
y = 3*x**2 + 6*x + 5

ax.plot(x, y)
plt.grid()
plt.show()
pass

