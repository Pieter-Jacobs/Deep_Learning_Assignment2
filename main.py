

from imports import *
from preprocess_data import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    dataset, train, test, val = load_data()
    labels = extract_labels(val)
    plot_hist(labels)
    embeddings = compute_embeddings([ex['text'] for ex in list(val)])
    plot_scatter(embeddings, labels)
    pass

if __name__ == '__main__':
    main()
