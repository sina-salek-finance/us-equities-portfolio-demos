from matplotlib import pyplot as plt


def plot(xs, ys, labels, title="", x_label="", y_label=""):
    for x, y, label in zip(xs, ys, labels):
        # plt.ylim((0.5, 0.55))
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.show()
