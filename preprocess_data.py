from imports import *

def load_data():
    print("Loading dataset...")
    dataset = load_dataset('tweet_eval', 'sentiment')
    train, test, val = list(dataset.values())
    print(test)
    pass