from imports import *


def plot_scatter(X, y, filename):
    """Plot a 2D scatterplot"""
    sns.scatterplot(x=[ex[0] for ex in X], y=[
                    ex[1] for ex in X], palette=['y', 'g', 'r'], hue=y, style=y).set(title='Text embeddings')
    plt.legend(["Negative", "Neutral", "Positive"])
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(
        f"{hydra.utils.get_original_cwd()}{os.sep}img{os.sep}data_distribution_{filename}.pdf")
    plt.close()


def plot_hist_and_get_counts(y, filename):
    """Plot vertical histogram with count values"""
    g = sns.displot(y=y, discrete=True, legend=False,
                    shrink=0.8, palette=['g', 'y', 'r'], hue=y, linewidth=0)
    plt.legend(["Negative", "Neutral", "Positive"])
    plt.yticks([], [])
    counts = annotate_bars(g)
    plt.savefig(
        f"{hydra.utils.get_original_cwd()}{os.sep}img{os.sep}class_distribution_{filename}.pdf")
    plt.close()
    return counts


def annotate_bars(plot):
    """Annotate the bars of a histogram with their count"""
    counts = []
    for ax in plot.axes.ravel():
        for p in ax.patches[2:7:2]:
            counts.append(p.get_width())
            ax.annotate(text=p.get_width(), xy=(
                p.get_width() + 1, p.get_y() + p.get_height()/2))
    return counts
