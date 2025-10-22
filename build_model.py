from collections import Counter

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn as nn

# Load data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Check for class imbalance and augment data if necessary
def check_data(data, imbalance_threshold= 0.8):
    columns = data.columns[1:]
    for col in columns:
        counter = Counter(data[col])

    if len(counter) > 1:
        max_count = max(counter[1:].values())
        min_count = min(v for v in counter[1:].values() if v > 0)
        ratio = round(min_count / max_count, 2)
    
    if ratio < imbalance_threshold:
        data = Augment_data(data, col, counter)

    return data    

# Augment data to balance classes
def Augment_data(data, col, counter):
    max_count = max(counter[1:].values())
    for class_label, count in counter.items():
        if class_label == 0:
            continue
        while count < max_count:
            sample = data[data[col] == class_label].sample(n=1, replace=True)
            data = pd.concat([data, sample], ignore_index=True)
            count += 1
    return data  

def tokenize_data(tokenizer, texts, max_length=512):
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="np"
    )
    return encodings

class MultiLabelDataLoader(nn.Module):

    def __init__(self, tokenizer, model_name, dropout, num_aspects, num_segmments):
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.num_aspects = num_aspects
        self.num_segmments = num_segmments
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = dropout
        self.aspects_classificer = nn.Linear(self.hidden_size, self.num_aspects)
        self.segments_classificer = nn.Linear(self.hidden_size, self.num_segmments + self.num_aspects)
    
    def forward(self):
        out = self.encoder(**self.tokenizer, return_dict=True)
        pooled = out.pooler_output if hasattr(out, 'pooler_output') and out.pooler_output is not None else out.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        aspects_logits = self.aspects_classificer(pooled)
        segments_logits = self.segments_classificer(pooled)

        return aspects_logits, segments_logits