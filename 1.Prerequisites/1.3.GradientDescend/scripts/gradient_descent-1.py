import time
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axisartist.axislines import AxesZero


# Define the functions for f(x) and its gradient
def fx(x):
    return 3*x**2 + 6*x + 5

def fx_gd(x):
    return 6*x + 6

def line(x, y, k):
    b = y - k*x
    p1_x = 5
    p1_y = k*p1_x + b
    p2_y = 0
    p2_x = -b/(k+1e-5)
    
    return [p1_x, p2_x], [p1_y, p2_y]

# Define a function that updates the plot with the current x and y values
def update_plot(i):
    global x, y
    gradient = fx_gd(x)
    x_new = x - lr * gradient
    y_new = fx(x_new)
    ax.clear()
    for direction in ['xzero', 'yzero']:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)
    for direction in ['left', 'right', 'bottom', 'top']:
        ax.axis[direction].set_visible(False)
        
    ax.set_xlim((-5, 5))
    ax.plot(xticks, f)
    ax.plot(x, y, marker='o')
    
    line_x, line_y = line(x_new, y_new, gradient)
    ax.plot(line_x, line_y)
    
    x_change = abs(x_new - x)
    if x_change < tolerance:
        ani.event_source.stop()
        print(f"Stopped after {i} iterations")
        exit()

    x = x_new
    y = y_new
    print(f"Iteration: {i}, Currently x={x:.4f}, y={y:.4f}, error: {x_change}")
    time.sleep(0.1)

if __name__ == '__main__':
    # Create the figure and the axes
    fig = plt.figure()
    ax = fig.add_subplot(axes_class=AxesZero)

    # Set the x-axis limits and create the x-values for the function
    xticks = np.linspace(-5, 5, 100)
    f = fx(xticks) # Create the function values for the x-axis values
    x = 5          # Set the initial value of x
    y = fx(x)      # Set the initial value of y

    lr = 0.01        # Set the learning rate
    epochs = 1000    # Set the maximum number of iterations
    tolerance = 1e-4 # Set the tolerance level for convergence

    # Create a FuncAnimation object to animate the plot
    ani = FuncAnimation(fig, update_plot, frames=epochs, interval=50)

    # Show the plot
    plt.grid()
    plt.show()


