from imports import *
from preprocess_data import *
from bert_from_scratch import *

# Make results reproducible
SEED = 1815
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def init_iterators(cfg, train_ds, val_ds, test_ds, device):
    train_iterator = data.BucketIterator(
        dataset=train_ds
        batch_size=cfg.parameters.batch_size,
        device=device
    )
    val_iterator = data.BucketIterator(
        dataset=val_ds,
        batch_size=cfg.parameters.batch_size,
        device=device
    )
    test_iterator = data.BucketIterator(
        dataset=test_ds,
        batch_size=cfg.parameters.batch_size,
        device=device
    )

@hydra.main(config_path=os.getcwd(), config_name="config.yaml")
def main(cfg: DictConfig):
    device = torch.device('cpu')
    dataset, train, test, val = load_data()
    model = load_untrained_bert()
    
    # labels = extract_labels(val)
    # plot_hist(labels)
    # embeddings = compute_embeddings([ex['text'] for ex in list(val)])
    # plot_scatter(embeddings, labels)
    pass

if __name__ == '__main__':
    main()
