from imports import *
import numpy as np

def load_untrained_bert():
    config = transformers.AutoConfig.from_pretrained('bert-base-uncased')
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
    model = transformers.AutoModel.from_config(config)
    return model


def train(model, optimizer, criterion, cfg):
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


def validation_step(model, iterator):
    """"
    Evaluates the performance of the model on the validation set

    Returns:
    (epoch_acc / len(self.val_iterator)): float
        Average validation accuracy over the different batches
    (epoch_loss / len(self.val_iterator)): float
        Average validation loss over the different batches
    """
    epoch_acc = 0
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.token_ids,
                                         token_type_ids=None,
                                         attention_mask=batch.mask,
                                         labels=batch.label.long())
            loss = predictions[0]
            acc = compute_accuracy(predictions[1], batch.label.long())
            epoch_acc += acc
            epoch_loss += float(loss.item())
    return (epoch_acc / len(iterator)), (epoch_loss / len(iterator))


def training_step(model, iterator, optimizer):
    """
    Trains the model based on one pass through all data
    Returns:
    --------
    epoch_acc / len(self.train_iterator): float
        Average training accuracy over the different batches
    epoch_loss / len(self.train_iterator): float
        Average training loss over the different batches
    """
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.token_ids,
                                     token_type_ids=None,
                                     attention_mask=batch.mask,
                                     labels=batch.label.long())
        loss = predictions[0]
        acc = compute_accuracy(predictions[1], batch.label.long())
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.item())
        epoch_acc += float(acc)
    return epoch_acc / len(iterator), epoch_loss / len(train_iterator)

def compute_accuracy(self, predictions, y):
    """Computes the accuracy of the made predictions, so it can be printed"""
    correct = 0.0
    for i, pred in enumerate(predictions):
        # Check if prediction is the same as supervised label
        if torch.tensor(np.argmax(pred.detach().cpu().numpy())) == y[i].cpu():
            correct += 1
    return correct / len(predictions)


def compute_f1():
    pass