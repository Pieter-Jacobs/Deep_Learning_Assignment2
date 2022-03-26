from imports import *
import numpy as np


tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')

def load_data():
    print("Loading dataset...")
    dataset_split = datasets.load_dataset('tweet_eval', 'sentiment')
    dataset_split = dataset_split.map(tokenize, batched=True)
    dataset_split = dataset_split.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset_split.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    dataset_full = datasets.concatenate_datasets(list(dataset_split.values()))
    train, test, val = list(dataset_split.values())
    return dataset_full, train, test, val

def extract_labels(dataset):
    labels_int = [ex['label'] for ex in list(dataset)]
    labels = dataset.features['label'].int2str(labels_int)
    capitalized_labels = [label.capitalize() for label in labels]
    return capitalized_labels

def plot_hist(y):
    """Plot vertical histogram with count values"""
    g = sns.displot(y=y, discrete=True, legend=False,
                    shrink=0.8, palette=['g', 'y', 'r'], hue=y, linewidth=0)
    annotate_bars(g)
    plt.show()
    # plt.savefig(f"img{os.sep}class_distribution.pdf")
    #plt.close()

def plot_scatter(X, y):
    """Plot a 2D scatterplot"""
    sns.scatterplot(x=[ex[0] for ex in X], y=[
                    ex[1] for ex in X], palette=['g', 'y', 'r'], hue=y).set(title='Data distribution')
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()
    plt.close()

def compute_embeddings(texts):
    print("Computing embeddings...")
    pca = PCA(n_components = 2)
    embeddings = []
    sentenceTransformer = SentenceTransformer(
        'all-mpnet-base-v2')
    for text in texts:
        embedding = np.zeros(768)
        for sentence in tokenize.sent_tokenize(text):
            embedding += sentenceTransformer.encode(
                sentence, show_progress_bar=False)
        embeddings.append(embedding)
    embeddings = pca.fit_transform(embeddings)
    return embeddings

def annotate_bars(plot):
    """Annotate the bars of a histogram with their count"""
    for ax in plot.axes.ravel():
        for p in ax.patches[2:7:2]:
            ax.annotate(text=p.get_width(), xy=(
                p.get_width() + 1, p.get_y() + p.get_height()/2))

def create_fields():
    pass

def tokenize(example):
    return tokenizer(example['text'], padding='max_length')