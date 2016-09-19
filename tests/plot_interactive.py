import tmgen
from matplotlib import pyplot as plt
from tmgen.plot import heatmap

plt.figure()
tm = tmgen.exp_tm(10, 500)  # 10 nodes, mean of 500 flows per IE
heatmap(tm, annot=True)
plt.show()  # display the graph interactively