from imports import *

def load_data():
    print("Loading dataset...")
    dataset_split = datasets.load_dataset('tweet_eval', 'sentiment')
    dataset_full = datasets.concatenate_datasets(list(dataset_split.values()))
    train, test, val = list(dataset_split.values())
    return dataset_full, train, test, val

def plot_hist(dataset):
    """Plot vertical histogram with count values"""
    labels_int = [ex['label'] for ex in list(dataset)]
    labels = dataset.features['label'].int2str(labels_int)
    capitalized_labels = [label.capitalize() for label in labels]
    g = sns.displot(y=capitalized_labels, discrete=True, legend = False,
                 shrink=0.8, palette=['g', 'y', 'r'], hue=capitalized_labels, linewidth=0)
    #plt.xticks(ticks=,labels=class_labels)
    #annotate_bars(g)
    plt.show()
    # plt.savefig(f"img{os.sep}class_distribution.pdf")
    #plt.close()

def annotate_bars(plot):
    """Annotate the bars of a histogram with their count"""
    for ax in plot.axes.ravel():
        for p in ax.patches[1:3]:
            ax.annotate(text=p.get_width(), xy=(
                p.get_width() + 1, p.get_y() + p.get_height()/2))
