from imports import *
import numpy as np


def load_untrained_bert():
    config = transformers.BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 3
    # config = transformers.BertConfig(
    # vocab_size=2048,
    # max_position_embeddings=768,
    # intermediate_size=2048,
    # hidden_size=512,
    # num_attention_heads=8,
    # num_hidden_layers=6,
    # type_vocab_size=5,
    # hidden_dropout_prob=0.1,
    # attention_probs_dropout_prob=0.1,
    # num_labels=3,
    # )
    model = transformers.BertForSequenceClassification(config)
    return model

def load_trained_bert():
    model = transformers.BertForSequenceClassification.from_pretrained(
        # Use the 12-layer BERT model, with an uncased vocab.
        "bert-base-uncased",
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False,
    )
    return model

def train(model, optimizer, cfg, train_dataloader, val_dataloader, device):
    """
    Trains the model for the chosen amount of epochs using early stopping
    Parameters:
    -----------
    """
    loss = []
    acc = []
    val_losses = []
    print("Starting training...")
    for epoch in range(cfg.n_epochs):
        print("Epoch number: " + str(epoch))
        torch.save(model.state_dict(
        ), f"{hydra.utils.get_original_cwd()}{os.path.sep}saves{os.path.sep}model_early_stopping_{str(epoch)}.pkl")

        train_acc, train_loss = training_step(
            model=model, dataloader=train_dataloader, optimizer=optimizer, device=device)
        val_acc, val_loss = validation_step(
            model=model, dataloader=val_dataloader, device=device)

        print(
            f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(
            f'\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%')

        loss.append(train_loss)
        acc.append(train_acc)
        val_losses.append(val_loss)

        # Load the parameters of the model with the lowest validation loss
        model.load_state_dict(torch.load(
            f"{hydra.utils.get_original_cwd()}{os.path.sep}saves{os.path.sep}model_early_stopping_{str(epoch)}.pkl"))
        optimizer.params = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, eps=cfg.eps)


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
        print(batch['labels'])
        acc = compute_accuracy(predictions[1], batch['labels'].long())
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
            if(i == 50):
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(**batch)
            loss = predictions[0]
            acc = compute_accuracy(predictions[1], batch['labels'].long())
            average_acc += acc
            epoch_loss += float(loss.item())
    return (average_acc / len(dataloader)), (epoch_loss / len(dataloader))


def compute_accuracy(predictions, y):
    """Computes the accuracy of the made predictions, so it can be printed"""
    correct = 0.0
    for i, pred in enumerate(predictions):
        # Check if prediction is the same as supervised label
        if torch.tensor(np.argmax(pred.detach().cpu().numpy())) == y[i].cpu().item():
            correct += 1
    return correct / len(predictions)


def compute_f1():
    pass

def evaluate(model, dataloader):
    """
    Makes evaluation steps corresponding to the amount of epochs and prints the loss and accuracy
    Returns:
    avg_loss: float
        Average loss of the model predictions
    """
    model.eval()
    average_acc = 0
    average_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            predictions = model(batch.token_ids,
                                        token_type_ids=None,
                                        attention_mask=batch.mask,
                                        labels=batch.label.long())
            loss = predictions[0]
            batch_acc = compute_accuracy(
                predictions[1], batch.label.long())
            average_acc += batch_acc
            average_loss += float(loss.item())

    f = open(f"{hydra.utils.get_original_cwd()}{os.path.sep}accuracy.txt", "a")
    f.write(str(average_acc / len(dataloader)))
    f.write(" ")
    f.close()

    return (average_acc / len(dataloader)), (average_loss / len(dataloader))

