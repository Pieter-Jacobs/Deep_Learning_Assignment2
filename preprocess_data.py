from imports import *

def load_data():
    print("Loading dataset...")
    dataset_split = datasets.load_dataset('tweet_eval', 'sentiment')
    dataset_full = datasets.concatenate_datasets(list(dataset_split.values()))
    train, test, val = list(dataset_split.values())
    pass

