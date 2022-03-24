from imports import *
from preprocess_data import *
from bert_from_scratch import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@hydra.main(config_path=os.getcwd(), config_name="config.yaml")
def main(cfg: DictConfig):
    dataset, train, test, val = load_data()
    model = load_untrained_bert()
    # labels = extract_labels(val)
    # plot_hist(labels)
    # embeddings = compute_embeddings([ex['text'] for ex in list(val)])
    # plot_scatter(embeddings, labels)
    pass

if __name__ == '__main__':
    main()
