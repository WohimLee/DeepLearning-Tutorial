
import numpy as np
import matplotlib.pyplot as plt

def normalize(x, y):
    x_mean = np.mean(x)
    x_std  = np.std(x)
    y_mean = np.mean(y)
    y_std  = np.std(y)
    x = (x - x_mean) / x_std
    y = (y - y_mean) / y_std
    return x, y


if __name__ == '__main__':
    
    x = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019], dtype=np.float32)
    y = np.array([1.8, 2.1, 2.3, 2.3, 2.85, 3.0, 3.3, 4.9, 5.45, 5.0], dtype=np.float32)
    
    
    x_norm, y_norm = normalize(x, y)
    
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the data on the first subplot
    axs[0].scatter(x, y)
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Price')
    axs[0].set_title('House Price')
    axs[0].grid()

    axs[1].scatter(x_norm, y_norm)
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Price')
    axs[1].set_title('House Price - Normalize')
    axs[1].grid()


    # Display the figure
    plt.show()
    # plt.savefig("./houseprice.png")
    