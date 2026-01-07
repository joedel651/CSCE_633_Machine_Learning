import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
import string
import re
from tqdm import tqdm
import math

# Download required NLTK data
nltk.download('punkt', quiet=True)

def preprocess_text(text):
    """
    Clean and tokenize text
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        return tokens
    return []

class Vocabulary:
    """
    Build a vocabulary from the word count
    """
    def __init__(self, max_size):
        self.max_size = max_size
        # Add <cls> token for transformer classification
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<cls>"}
        self.word_count = {}
        self.size = 3  # Start with pad, unk, and cls tokens
        
    def add_word(self, word):
        if word not in self.word_count:
            self.word_count[word] = 0
        self.word_count[word] += 1
            
    def build_vocab(self):
        sorted_words = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_words[:self.max_size - 3]
        
        for word, count in top_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                self.size += 1
                
    def text_to_indices(self, tokens, max_len, model_type='lstm'):
        if model_type == 'transformer':
            tokens = tokens[:max_len - 1]
            indices = [self.word2idx['<cls>']]
        else:
            tokens = tokens[:max_len]
            indices = []
        
        unk_idx = self.word2idx['<unk>']
        for token in tokens:
            if model_type == 'transformer' and token not in self.word2idx:
                continue
            indices.append(self.word2idx.get(token, unk_idx))
        
        padding_length = max_len - len(indices)
        indices.extend([self.word2idx['<pad>']] * padding_length)
        return indices

class IMDBDataset(Dataset):
    """
    A dataset for LSTM (returns 2 items)
    """
    def __init__(self, dataframe, vocabulary, max_len, is_training=True, model_type='lstm'):
        self.dataframe = dataframe
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.model_type = model_type
        self.processed_data = []
        
        for i in range(len(dataframe)):
            text = dataframe.iloc[i]['text']
            tokens = preprocess_text(text)
            label = dataframe.iloc[i]['label']
            
            indices = vocabulary.text_to_indices(tokens, max_len, model_type)
            attention_mask = [1 if idx != vocabulary.word2idx['<pad>'] else 0 for idx in indices]
            
            # Store all 3 for flexibility
            item = (indices, attention_mask, label)
            self.processed_data.append(item)
            
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        indices, attention_mask, label = self.processed_data[idx]
        
        # --- FIX: Conditionally return 2 or 3 items ---
        input_tensor = torch.tensor(indices, dtype=torch.long)
        label_tensor = torch.tensor([label], dtype=torch.float)  # Float for BCEWithLogitsLoss

        if self.model_type == 'lstm':
            # LSTM needs input_ids and label (2 items)
            return (input_tensor, label_tensor)
        else:
            # Transformer needs input_ids, attention_mask, and label (3 items)
            attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
            return (input_tensor, attention_mask_tensor, label_tensor)
        
# LSTM model
class LSTM(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=256, output_dim=1,
                 n_layers=2, bidirectional=True, dropout=0.5, pad_idx=0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids shape: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(input_ids))
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        output, (hidden, cell) = self.lstm(embedded)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        # output shape: (batch_size, seq_len, hidden_dim * num_directions)
        
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            # hidden[-2] is the last layer's forward hidden state
            # hidden[-1] is the last layer's backward hidden state
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            # Just use the last layer's hidden state
            hidden = hidden[-1]
        
        # hidden shape: (batch_size, hidden_dim * num_directions)
        hidden = self.dropout(hidden)
        return self.fc(hidden)
    
# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size=10000, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=512, output_dim=1, dropout=0.1, pad_idx=0, max_len=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        embedded = self.dropout(embedded)
        
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        
        cls_output = transformer_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        return self.fc(cls_output)

def load_and_preprocess_data(data_path, data_type='train', model_type='lstm', shared_vocab=None):
    """
    Load and preprocess the IMDB dataset
    
    Args:
        data_path: Path to the data files
        data_type: Type of data to load ('train' or 'test')
        model_type: Type of model ('lstm' or 'transformer')
        shared_vocab: Optional vocabulary to use (for test data)
    
    Returns:
        data_loader: DataLoader for the specified data type
        vocab: Vocabulary object (only returned for train data)
    """
    # Fixed hyperparameters
    test_size = 0.2
    random_state = 42
    vocab_size = 10000
    max_len = 512
    batch_size = 32
    
    # Load full dataset
    df_full = pd.read_parquet(data_path)
    
    # Handle train/val split
    if data_type == 'train':
        df_train, df_val = train_test_split(df_full, test_size=test_size, 
                                            random_state=random_state, 
                                            stratify=df_full['label'])
        df = df_train.reset_index(drop=True)
    elif data_type == 'val':
        df_train, df_val = train_test_split(df_full, test_size=test_size, 
                                            random_state=random_state, 
                                            stratify=df_full['label'])
        df = df_val.reset_index(drop=True)
    else:
        # For test or any other data_type, use full dataframe
        df = df_full.reset_index(drop=True)
        df_train = df_full
    
    # Build or use existing vocabulary
    if data_type == 'train' and shared_vocab is None:
        vocab = Vocabulary(max_size=vocab_size)
        for i in range(len(df_train)):
            text = df_train.iloc[i]['text']
            tokens = preprocess_text(text)
            for token in tokens:
                vocab.add_word(token)
        vocab.build_vocab()
    else:
        vocab = shared_vocab
    
    if vocab is None:
        raise ValueError("Vocabulary must be provided for non-train data")
    
    # Create dataset - it handles returning correct format based on model_type
    dataset = IMDBDataset(df, vocab, max_len, is_training=(data_type == 'train'), model_type=model_type)
    
    # Create dataloader with default collate
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=(data_type == 'train'))
    
    # Return based on data_type
    if data_type == 'train':
        return (data_loader, vocab)
    else:
        return data_loader

def train(model, iterator, optimizer, criterion, device, model_type='lstm'):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        # --- FIX: Conditionally unpack 2 or 3 items ---
        if model_type == 'transformer':
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
        else:
            # For LSTM (2 items)
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
        # --- END FIX ---
        
        optimizer.zero_grad()
        
        # Use attention_mask only for transformer
        if model_type == 'transformer':
            predictions = model(input_ids, attention_mask)
        else:
            predictions = model(input_ids)
        
        loss = criterion(predictions, labels)
        acc = ((predictions > 0).float() == labels).float().mean()
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device, model_type='lstm'):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in iterator:
            # --- FIX: Conditionally unpack 2 or 3 items ---
            if model_type == 'transformer':
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
            else:
                # For LSTM (2 items)
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
            # --- END FIX ---
            
            # Use attention_mask only for transformer
            if model_type == 'transformer':
                predictions = model(input_ids, attention_mask)
            else:
                predictions = model(input_ids)
            
            loss = criterion(predictions, labels)
            acc = ((predictions > 0).float() == labels).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    MAX_LEN = 512
    VOCAB_SIZE = 10000
    BATCH_SIZE = 32
    EPOCHS = 10
    
    # Choose model type
    MODEL_TYPE = 'transformer'  # or 'transformer'
    
    # Load data and split into train/validation
    print("Loading data...")
    df_full = pd.read_parquet('train.parquet')
    
    # Split into train and validation (80/20 split)
    df_train, df_val = train_test_split(df_full, test_size=0.2, random_state=42, stratify=df_full['label'])
    
    # Build vocabulary on training data only
    print("Building vocabulary...")
    vocab = Vocabulary(max_size=VOCAB_SIZE)
    for i in tqdm(range(len(df_train)), desc="Building vocab"):
        text = df_train.iloc[i]['text']
        tokens = preprocess_text(text)
        for token in tokens:
            vocab.add_word(token)
    vocab.build_vocab()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = IMDBDataset(df_train.reset_index(drop=True), vocab, MAX_LEN, is_training=True, model_type=MODEL_TYPE)
    val_dataset = IMDBDataset(df_val.reset_index(drop=True), vocab, MAX_LEN, is_training=False, model_type=MODEL_TYPE)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Vocabulary size: {vocab.size}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    if MODEL_TYPE == 'lstm':
        model = LSTM(
            vocab_size=vocab.size,
            embedding_dim=100,
            hidden_dim=256,
            output_dim=1,
            n_layers=2,
            bidirectional=True,
            dropout=0.5,
            pad_idx=vocab.word2idx['<pad>']
        )
    else:  # transformer
        model = TransformerEncoder(
            vocab_size=vocab.size,
            d_model=128,
            nhead=8,
            num_layers=3,
            dim_feedforward=512,
            output_dim=1,
            dropout=0.1,
            pad_idx=vocab.word2idx['<pad>'],
            max_len=MAX_LEN
        )
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nTraining {MODEL_TYPE.upper()} model...")
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Training with progress bar
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            if MODEL_TYPE == 'transformer':
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
            else:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if MODEL_TYPE == 'transformer':
                predictions = model(input_ids, attention_mask)
            else:
                predictions = model(input_ids)
            
            loss = criterion(predictions, labels)
            acc = ((predictions > 0).float() == labels).float().mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_acc += acc.item()
        
        train_loss = epoch_train_loss / len(train_loader)
        train_acc = epoch_train_acc / len(train_loader)
        
        # Evaluation with progress bar
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                if MODEL_TYPE == 'transformer':
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                else:
                    input_ids, labels = batch
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                
                if MODEL_TYPE == 'transformer':
                    predictions = model(input_ids, attention_mask)
                else:
                    predictions = model(input_ids)
                
                loss = criterion(predictions, labels)
                acc = ((predictions > 0).float() == labels).float().mean()
                
                epoch_val_loss += loss.item()
                epoch_val_acc += acc.item()
        
        val_loss = epoch_val_loss / len(val_loader)
        val_acc = epoch_val_acc / len(val_loader)
        
        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{MODEL_TYPE}.pt')
            print(f"Saved best model with validation accuracy: {val_acc*100:.2f}%")
    
    print(f"\nBest validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved as {MODEL_TYPE}.pt")

if __name__ == "__main__":
    main()
