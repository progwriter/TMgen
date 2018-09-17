import tmgen
from matplotlib import pyplot as plt
from tmgen.plot import heatmap

if __name__ == '__main__':
    plt.figure()
    tm = tmgen.models.exp_tm(10, 500)  # 10 nodes, mean of 500 flows per IE
    heatmap(tm)
    plt.show()  # display the graph interactively
