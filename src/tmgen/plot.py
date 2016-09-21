try:
    import seaborn
    seaborn.set(style='white')
except ImportError:
    import warnings
    warnings.warn('Consider installing searbon package for prettier graphs',
                  category=ImportError)
from matplotlib import pyplot as plt
from six.moves import range



def heatmap(tm, epoch=None):
    """
    Plot the traffic matrix as a heatmap using matplotlib.
    If epoch is given or traffic matrix has a single epoch then the graph is a
    heatmap depicting flow volume between each ingress-egress pair.

    If tm has multiple epochs, the graph will have ingress-egress pairs
    enumerated sequentially on the y-axis and epochs on x-axis.

    :param tm: the traffic matrix to visualize
    :param epoch: the epoch to plot. If None, all epochs will be plotted

    .. note::

        This uses the matplotlib.pyplot for graphing. It is the user's
        responsibility to create new figures and display (or save) the plot

    Example: ::

        import tmgen
        from matplotlib import pyplot as plt
        from tmgen.plot import heatmap

        plt.figure()
        tm = tmgen.exp_tm(10, 500) # 10 nodes, mean of 500 flows per IE
        tmgen.plot.heatmap(tm)
        plt.show()  # display the graph interactively
    """
    n = tm.num_pops()
    reshaped = False
    if epoch is not None:
        matrix = tm.at_time(epoch)
    else:
        if tm.num_epochs() == 1:
            matrix = tm.at_time(0)
        else:
            matrix = tm.matrix.reshape((n * n, tm.num_epochs()))
            reshaped = True
    if not reshaped:
        seaborn.heatmap(matrix)
        plt.xlabel('Node')
        plt.ylabel('Node')
        plt.title('Volume at epoch {}'.format(epoch if epoch is not None
                                              else 0))
    else:
        seaborn.heatmap(matrix, xticklabels=max(tm.num_epochs() / 10, 1),
                        yticklabels=n)
        plt.xlabel('Time (epochs)')
        plt.ylabel('Ingress-egress pair')


def timeseries(tm):
    """
    Plot the traffic matrix as timeseries. With epochs on the x-axis and volume
    of the y-axis each line represents a single OD pair.

    :param tm: the traffic matrix to visualize

    .. note::

        This uses the matplotlib.pyplot for graphing. It is the user's
        responsibility to create new figures and display (or save) the plot

    Example: ::

        import tmgen
        from matplotlib import pyplot as plt
        from tmgen.plot import heatmap

        plt.figure()
        tm = tmgen.exp_tm(10, 500) # 10 nodes, mean of 500 flows per IE
        tmgen.plot.timeseries(tm)
        plt.show()  # display the graph interactively

    """
    n = tm.num_pops()
    for i in range(n):
        for e in range(n):
            plt.plot(tm.between(i, e, 'all'))
    plt.xlabel('Epoch')
    plt.ylabel('Volume')
