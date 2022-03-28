from imports import *
import numpy as np


tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
print(dir(tokenizer))

def s140_to_tweet_eval(label):
    label_mapping = {
        0: 0,
        2: 1,
        4: 2
    }
    return label_mapping[label]


def load_tweet_eval():
    dataset = datasets.load_dataset('tweet_eval', 'sentiment')
    dataset = dataset.rename_column('label', 'labels')
    dataset = datasets.concatenate_datasets(list(dataset.values()))
    print(dataset[0])
    print(dataset)
    return dataset

def load_s140():
    print("Loading dataset...")
    s140 = datasets.load_dataset('sentiment140')
    s140 = s140.map(lambda examples: {'labels': [
                                      s140_to_tweet_eval(label) for label in examples['sentiment']]}, batched=True)
    s140 = s140.remove_columns(
        ["sentiment", "user", "date", "query"])
    dataset = datasets.concatenate_datasets(list(s140.values()))
    return dataset

def train_test_val_split(dataset):
    train_test = dataset.train_test_split(test_size=0.1)
    test_valid = train_test['test'].train_test_split(test_size=0.5)
    dataset = datasets.DatasetDict({
        'train': train_test['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})
    return dataset

def generalise_dataset(dataset):
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=[
                       'input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    return dataset

def plot_hist_and_get_counts(y):
    """Plot vertical histogram with count values"""
    g = sns.displot(y=y, discrete=True, legend=False,
                    shrink=0.8, palette=['g', 'y', 'r'], hue=y, linewidth=0)
    plt.legend(["Negative", "Neutral", "Positive"])
    plt.yticks([], [])
    counts = annotate_bars(g)
    plt.savefig(
        f"{hydra.utils.get_original_cwd()}{os.sep}img{os.sep}class_distribution.pdf")
    plt.show()
    plt.close()
    return counts


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
    pca = PCA(n_components=2)
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
    counts = []
    for ax in plot.axes.ravel():
        for p in ax.patches[2:7:2]:
            counts.append(p.get_width())
            ax.annotate(text=p.get_width(), xy=(
                p.get_width() + 1, p.get_y() + p.get_height()/2))
    return counts


def tokenize(example):
    return tokenizer(example['text'], padding='max_length')
