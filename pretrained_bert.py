# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:47:16 2022

@author: usuario
"""

from imports import *
import preprocess_data as preprocess
from bert_from_scratch import *


@hydra.main(config_path=os.getcwd(), config_name="config.yaml")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, train_ds, test_ds, val_ds = preprocess.load_data()
    model = load_trained_bert()
    train_dataloader, val_dataloader, test_dataloader = init_dataloaders(
      cfg, train_ds, test_ds, val_ds, device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=cfg.lr, eps=cfg.eps)
    training_process(model, optimizer, cfg, train_dataloader, val_dataloader, device)

def training_process(model, optimizer, cfg, train_dataloader, val_dataloader, device):
    
    epochs = 4
    
    # TRAINING
    for epoch in range(epochs):
        
        model = model.train()
        
        for i, batch in enumerate(dataloader):
            
            optimizer.zero_grad()
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
    
            label = contents_labels_train[index]
            loss, logits = model(b_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask, 
                                 labels=b_labels)
            
            total_train_loss += loss.item()
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
        avg_train_loss = total_train_loss / len(train_dataloader) 
        
        
        # VALIDATION
        
        model = model.eval()
    
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
    
        for batch in validation_dataloader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        
                
                (loss, logits) = model(b_input_ids, 
                                       token_type_ids=None, 
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()
    
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
    
            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            
    
        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
    
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
        
        # Display floats with two decimal places.
    pd.set_option('precision', 2)
    
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')
    
    # A hack to force the column headers to wrap.
    #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
    
    # Display the table.
    df_stats
    
    
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')
    
    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)
    
    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    
    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])
    
    plt.show()
    
    
def testing_process(model, test_dataloader, device):
    
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []
    
    # Predict 
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
      
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
      
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)
    
        logits = outputs[0]
    
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
      
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
      
        
        
        
        
        
        
        
