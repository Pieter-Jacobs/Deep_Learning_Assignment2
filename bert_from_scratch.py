from imports import *
import numpy as np


def write_to_file(folder, file, text):
    f = open(f"{hydra.utils.get_original_cwd()}{os.path.sep}evaluation_data{os.path.sep}{folder}{os.path.sep}{file}", "a")
    f.write(text)
    if "\n" not in text:
        f.write(" ")
    f.close()


def load_untrained_bert(dropout):
    config = transformers.BertConfig(
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        num_labels=3,
    )
    model = transformers.BertForSequenceClassification(config)
    return model


def load_pretrained_bert(dropout):
    model = transformers.BertForSequenceClassification.from_pretrained(
        # Use the 12-layer BERT model, with an uncased vocab.
        "bert-base-uncased",
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout
    )
    return model


def train(model, train_dataloader, val_dataloader, optimizer, device, epochs, pretrained, dropout):
    """
    Trains the model for the chosen amount of epochs using early stopping
    Parameters:
    -----------
    """
    val_losses = []

    print("Starting training...")
    for epoch in range(epochs):
        print("Epoch number: " + str(epoch))
        torch.save(model.state_dict(
        ), f"{hydra.utils.get_original_cwd()}{os.path.sep}saves{os.path.sep}model_early_stopping_{str(epoch)}.pkl")

        train_acc, train_loss = training_step(
            model=model, dataloader=train_dataloader, optimizer=optimizer, device=device)
        val_acc, val_loss = validation_step(
            model=model, dataloader=val_dataloader, device=device)
        val_losses.append(val_loss)
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(
            f'\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%')

        folder = "pretrained" if pretrained else "untrained"
        folder = folder + \
            f"{os.sep}normal" if dropout == 0 else folder + f"{os.sep}dropout"

        write_to_file(folder, "train_accuracy.txt", str(train_acc))
        write_to_file(folder, "train_loss.txt", str(train_loss))
        write_to_file(folder, "val_accuracy.txt", str(val_acc))
        write_to_file(folder, "val_loss.txt", str(val_loss))

    # Load the parameters of the model with the lowest validation loss
    print(str(np.argmin(val_losses)))
    model.load_state_dict(torch.load(
        f"{hydra.utils.get_original_cwd()}{os.path.sep}saves{os.path.sep}model_early_stopping_{str(np.argmin(val_losses))}.pkl"))
    optimizer.params = torch.optim.AdamW(
        model.parameters())

    write_to_file(folder, "train_accuracy.txt", "\n")
    write_to_file(folder, "train_loss.txt", "\n")
    write_to_file(folder, "val_accuracy.txt", "\n")
    write_to_file(folder, "val_loss.txt", "\n")
    print("Training finished")

    return model


def training_step(model, dataloader, optimizer, device):
    """
    Trains the model based on one pass through all data
    Returns:
    --------
    average_acc / len(train_dataloader): float
        Average training accuracy over the different batches
    epoch_loss / len(train_dataloader): float
        Average training loss over the different batches
    """
    epoch_loss = 0
    average_acc = 0
    model.train()
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        predictions = model(**batch)
        loss = predictions[0]
        acc = accuracy_score(batch['labels'].cpu().detach().numpy(), np.argmax(
            predictions[1].cpu().detach().numpy(), axis=1))
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.item())
        average_acc += float(acc)
    return average_acc / len(dataloader), epoch_loss / len(dataloader)


def validation_step(model, dataloader, device):
    """"
    Evaluates the performance of the model on the validation set

    Returns:
    (average_acc / len(val_dataloader)): float
        Average validation accuracy over the different batches
    (epoch_loss / len(val_dataloader)): float
        Average validation loss over the different batches
    """
    average_acc = 0
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(**batch)
            loss = predictions[0]
            acc = accuracy_score(batch['labels'].cpu().detach().numpy(), np.argmax(
                predictions[1].cpu().detach().numpy(), axis=1))
            average_acc += acc
            epoch_loss += float(loss.item())
    return (average_acc / len(dataloader)), (epoch_loss / len(dataloader))


def evaluate(model, dataloader, device, pretrained, dropout, T=None):
    """
    Makes evaluation steps corresponding to the amount of epochs and prints the loss and accuracy
    Returns:
    avg_loss: float
        Average loss of the model predictions
    """
    average_acc = 0
    average_acc_mcd = 0
    with torch.no_grad():
        for batch in dataloader:

            batch = {k: v.to(device) for k, v in batch.items()}

            model.eval()
            predictions = model(**batch)[1]
            batch_acc = accuracy_score(batch['labels'].cpu().detach().numpy(), np.argmax(
                predictions.cpu().detach().numpy(), axis=1))
            average_acc += batch_acc
            if dropout > 0:
                model = turn_on_dropout(model)
                predictions_mcd = np.argmax(np.array(
                    [model(**batch)[1].cpu().detach().numpy() for sample in range(T)]), axis=2)
                predictions_mcd = np.array(
                    stats.mode(predictions_mcd)[0])
                batch_acc_mcd = accuracy_score(
                    batch['labels'].cpu().detach().numpy(), predictions_mcd.flatten())
                average_acc_mcd += batch_acc_mcd

    folder = "pretrained" if pretrained else "untrained"
    folder = folder + \
        f"{os.sep}normal" if dropout == 0 else folder + f"{os.sep}dropout"
    write_to_file(folder, "test_accuracy.txt",
                  f"{average_acc / len(dataloader)}\n")

    if dropout > 0:
        write_to_file(folder, "test_accuracy_mcd.txt",
                      f"{average_acc_mcd / len(dataloader)}\n")
    return (average_acc / len(dataloader))


def turn_on_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.training = True   # Turn on dropout
    return model
