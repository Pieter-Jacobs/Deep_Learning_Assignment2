from imports import *
import preprocess_data as preprocess
from bert_from_scratch import *

# Make results reproducible
SEED = 1815
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def init_dataloaders(cfg, train_ds, val_ds, test_ds, device):
    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=cfg.batch_size
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_ds,
        batch_size=cfg.batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=cfg.batch_size
    )
    return train_loader, val_loader, test_loader


@hydra.main(config_path=os.getcwd(), config_name="config.yaml")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, train_ds, test_ds, val_ds = preprocess.load_data()
    model = load_untrained_bert()
    model = load_trained_bert()
    train_dataloader, val_dataloader, test_dataloader = init_dataloaders(
        cfg, train_ds, test_ds, val_ds, device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=cfg.lr, eps=cfg.eps)

    train(model=model, optimizer=optimizer, cfg=cfg, train_dataloader=train_dataloader, val_dataloader=val_dataloader, device=device)
    # labels = extract_labels(val)
    # plot_hist(labels)
    # embeddings = compute_embeddings([ex['text'] for ex in list(val)])
    # plot_scatter(embeddings, labels)
    pass


if __name__ == '__main__':
    main()
