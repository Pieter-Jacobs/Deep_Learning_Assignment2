from imports import *
from bert_helpers import *
import preprocess
import plot

# Make results reproducible
SEED = 1815
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@hydra.main(config_path=os.getcwd(), config_name="config.yaml")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data and plot the data class distribution
    dataset_tweet_eval, split = preprocess.load_tweet_eval()
    dataset_s140 = preprocess.load_s140()
    counts = plot.plot_hist_and_get_counts(
        dataset_tweet_eval['labels'], "tweet_eval")
    plot.plot_hist_and_get_counts(dataset_s140['labels'], "s140")

    # Extract positive and negative examples out of Sentiment140
    neg_s140 = dataset_s140.filter(lambda e: e['labels'] == 0)
    pos_s140 = dataset_s140.filter(lambda e: e['labels'] == 2)

    # Add data from Sentiment140 to TweetEval to balance classes
    imbalance = counts[np.argmax(counts)] - counts
    s140_for_balancing = datasets.concatenate_datasets([neg_s140.select(
        range(0, imbalance[2])), pos_s140.select(range(0, imbalance[0]))])
    dataset_tweet_eval = dataset_tweet_eval.cast(s140_for_balancing.features)
    dataset = datasets.concatenate_datasets(
        [dataset_tweet_eval, s140_for_balancing])

    # Compute and plot the text embeddings
    embeddings = preprocess.compute_embeddings(
        [ex['text'] for ex in list(dataset)])
    plot.plot_scatter(embeddings, dataset['labels'], "embeddings")

    # Prepare dataset to be used for training and testing
    dataset = preprocess.generalise_dataset(dataset)
    dataset = preprocess.train_test_val_split(dataset, split)
    train_loader, val_loader, test_loader = preprocess.init_dataloaders(
        dataset, cfg.batch_size)

    # Train and evaluate the model
    for run in range(cfg.runs):
        print(f"Run: {run}")
        model = load_pretrained_bert(
            cfg.dropout) if cfg.pretrained_bert else load_untrained_bert(cfg.dropout)
        model.to(device)
        optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=cfg.lr)
        model = train(model=model, train_dataloader=train_loader, val_dataloader=val_loader, optimizer=optimizer,
                      device=device, epochs=cfg.epochs, pretrained=cfg.pretrained_bert, dropout=cfg.dropout)
        evaluate(model=model, dataloader=test_loader, device=device, pretrained=cfg.pretrained_bert,
                 dropout=cfg.dropout, T=cfg.T)


if __name__ == '__main__':
    main()
