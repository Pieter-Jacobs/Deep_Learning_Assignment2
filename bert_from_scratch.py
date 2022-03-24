from imports import *


def load_untrained_bert():
    config = transformers.AutoConfig.from_pretrained('bert-base-uncased')
    model = transformers.AutoModel.from_config(config)
    return model


def train(model, X, y, optimizer, criterion, cfg):
    """
    Trains the model for the chosen amount of epochs using early stopping
    Parameters:
    -----------
    """
    loss = []
    acc = []
    epoch = 0
    val_losses = []
    print("Starting training...")
    for epoch in range(cfg.n_epochs):
        print("Epoch number: " + str(epoch))
        torch.save(model.state_dict(), os.path.join(
            "saves" + os.path.sep + 'model-early-stopping' + str(epoch) + '.pkl'))

        train_acc, train_loss = training_step()
        val_acc, val_loss = validation_step()

        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(
            f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%')

        loss.append(train_loss)
        acc.append(train_acc)
        val_losses.append(val_loss)

        # Load the parameters of the model with the lowest validation loss
        model.load_state_dict(torch.load(os.path.join(
            "saves" + os.path.sep + "model-early-stopping" + str(np.argmin(val_losses)) + ".pkl")), strict=False)
        optimizer.params = transformers.AdamW(
            model.parameters(), lr=cfg.lr, eps=cfg.eps)


def training_step():
    pass


def validation_step():
    pass
