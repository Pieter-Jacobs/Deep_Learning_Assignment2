from imports import *
import preprocess_data as preprocess
from bert_from_scratch import *

# Make results reproducible
SEED = 1815
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def init_dataloaders(dataset, batch_size):
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


@hydra.main(config_path=os.getcwd(), config_name="config.yaml")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_tweet_eval, split = preprocess.load_tweet_eval()
    dataset_s140 = preprocess.load_s140()

    counts = preprocess.plot_hist_and_get_counts(
        dataset_tweet_eval['labels'], "tweet_eval")
    neg_s140 = dataset_s140.filter(lambda e: e['labels'] == 0)
    pos_s140 = dataset_s140.filter(lambda e: e['labels'] == 2)
    preprocess.plot_hist_and_get_counts(dataset_s140['labels'], "s140")

    imbalance = counts[np.argmax(counts)] - counts
    s140_for_balancing = datasets.concatenate_datasets([neg_s140.select(
        range(0, imbalance[2])), pos_s140.select(range(0, imbalance[0]))])

    dataset_tweet_eval = dataset_tweet_eval.cast(s140_for_balancing.features)
    dataset = datasets.concatenate_datasets(
        [dataset_tweet_eval, s140_for_balancing])

    # embeddings = preprocess.compute_embeddings(
    #     [ex['text'] for ex in list(dataset)])
    # preprocess.plot_scatter(embeddings, dataset['labels'], "embeddings")

    dataset = preprocess.generalise_dataset(dataset)

    preprocess.plot_hist_and_get_counts(dataset['labels'], "generalised")

    dataset = preprocess.train_test_val_split(dataset, split)
    train_loader, val_loader, test_loader = init_dataloaders(
        dataset, cfg.batch_size)

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
