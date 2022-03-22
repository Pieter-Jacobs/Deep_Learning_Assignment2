

from imports import *
from preprocess_data import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    dataset, train, test, val = load_data()
    plot_hist(dataset)
    pass


if __name__ == '__main__':
    main()