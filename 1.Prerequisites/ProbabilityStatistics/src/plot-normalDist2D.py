
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import AxesZero

def normal2D(mean=[0, 0], cov= [[1, 0], [0, 1]]):
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
        
    # Generate random data from a 2D normal distribution
    # x = np.random.normal(loc=mean, scale=std, size=100)
    x, y = np.random.multivariate_normal(mean=mean, cov=cov, size=1000).T

    # Plot the data using matplotlib's scatter function
    ax.scatter(x, y, alpha=0.5)

    # Set plot labels and title
    ax.set_title('2D Normal Distribution')


    # Show the plot
    ax.grid()
    # plt.show()
    fig.savefig('../imgs/normalDist2D.png')

if __name__ == '__main__':
    normal2D()