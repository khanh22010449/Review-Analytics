import torch 
import torch.nn as nn
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, Tokenizer
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def tokenize_data(tokenizer, texts, max_length=512):
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

class mutil_label_classifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(mutil_label_classifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    mae = mean_absolute_error(y_true, y_pred)
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'mean_absolute_error': mae
    }

def train_model(model, train_dataloader, val_dataloader, epochs, optimizer, criterion, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss}")

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = compute_metrics(all_labels, all_preds)
        print(f"Validation Metrics: {metrics}")