from posixpath import split
from imports import *
import numpy as np
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')


def s140_to_tweet_eval(label):
    """Convert the Sentiment140 labels to TweetEval labels"""
    label_mapping = {
        0: 0,
        2: 1,
        4: 2
    }
    return label_mapping[label]


def load_tweet_eval():
    """Load the TweetEval dataset and return the original split ratio"""
    dataset = datasets.load_dataset('tweet_eval', 'sentiment')
    split = [dataset['train'].num_rows,
             dataset['test'].num_rows, dataset['validation'].num_rows]
    split = np.divide(split, np.sum(split))
    dataset = dataset.rename_column('label', 'labels')
    dataset = datasets.concatenate_datasets(list(dataset.values()))
    return dataset, split


def load_s140():
    """Load the Sentiment140 dataset"""
    print("Loading dataset...")
    s140 = datasets.load_dataset('sentiment140')
    s140 = s140.map(lambda examples: {'labels': [
                                      s140_to_tweet_eval(label) for label in examples['sentiment']]}, batched=True)
    s140 = s140.remove_columns(
        ["sentiment", "user", "date", "query"])
    dataset = datasets.concatenate_datasets(list(s140.values()))
    return dataset


def train_test_val_split(dataset, split):
    """Split the dataset into train, test and validation datasets"""
    train_test = dataset.train_test_split(test_size=split[1] + split[2])
    test_valid = train_test['test'].train_test_split(test_size=split[1])
    dataset = datasets.DatasetDict({
        'train': train_test['train'],
        'test': test_valid['train'],
        'validation': test_valid['test']})
    return dataset


def generalise_dataset(dataset):
    """Generalise the dataset to pytorch format"""
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=[
                       'input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    return dataset


def compute_embeddings(texts):
    """Compute the embeddings for a list of texts and reduce them to a 2D vector"""
    print("Computing embeddings...")
    pca = PCA(n_components=2)
    embeddings = []
    sentenceTransformer = SentenceTransformer(
        'all-mpnet-base-v2')
    for i, text in enumerate(texts):
        embedding = np.zeros(768)
        for sentence in nltk.tokenize.sent_tokenize(text):
            embedding += sentenceTransformer.encode(
                sentence, show_progress_bar=False)
        embeddings.append(embedding)
    embeddings = pca.fit_transform(embeddings)
    return embeddings


def tokenize(example):
    """Tokenize the input text"""
    return tokenizer(example['text'], padding='max_length')


def init_dataloaders(dataset, batch_size):
    """Initialise the dataloaders for the dataset splits"""
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset['train'],
        batch_size=batch_size
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset['validation'],
        batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset['test'],
        batch_size=batch_size
    )
    return train_loader, val_loader, test_loader
