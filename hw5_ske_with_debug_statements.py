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
        print(f"\n[Dataset __init__] Starting initialization")
        print(f"[Dataset __init__] model_type={model_type}")
        print(f"[Dataset __init__] dataframe shape={dataframe.shape}")
        print(f"[Dataset __init__] max_len={max_len}")
        
        self.dataframe = dataframe
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.model_type = model_type
        self.processed_data = []
        
        for i in range(len(dataframe)):
            if i < 3 or i >= len(dataframe) - 3:
                print(f"\n[Dataset __init__] Processing item {i}/{len(dataframe)}")
            
            text = dataframe.iloc[i]['text']
            tokens = preprocess_text(text)
            label = dataframe.iloc[i]['label']
            
            if i < 3:
                print(f"[Dataset __init__]   text length: {len(text)}")
                print(f"[Dataset __init__]   tokens: {tokens[:10] if len(tokens) > 10 else tokens}")
                print(f"[Dataset __init__]   label: {label}")
            
            indices = vocabulary.text_to_indices(tokens, max_len, model_type)
            
            if i < 3:
                print(f"[Dataset __init__]   indices length: {len(indices)}")
                print(f"[Dataset __init__]   indices[:10]: {indices[:10]}")
            
            attention_mask = [1 if idx != vocabulary.word2idx['<pad>'] else 0 for idx in indices]
            
            if i < 3:
                print(f"[Dataset __init__]   attention_mask length: {len(attention_mask)}")
            
            # Store all 3 for flexibility
            item = (indices, attention_mask, label)
            self.processed_data.append(item)
            
            if i < 3:
                print(f"[Dataset __init__]   Stored item with {len(item)} elements")
                print(f"[Dataset __init__]   Item types: {[type(x).__name__ for x in item]}")
        
        print(f"\n[Dataset __init__] COMPLETE")
        print(f"[Dataset __init__] Total items in processed_data: {len(self.processed_data)}")
        if len(self.processed_data) > 0:
            print(f"[Dataset __init__] First item length: {len(self.processed_data[0])}")
            print(f"[Dataset __init__] First item types: {[type(x).__name__ for x in self.processed_data[0]]}")
            print(f"[Dataset __init__] Last item length: {len(self.processed_data[-1])}")
        print()
            
    def __len__(self):
        length = len(self.processed_data)
        print(f"[__len__] Returning length: {length}")
        return length
    
    def __getitem__(self, idx):
        print(f"\n{'='*60}")
        print(f"[__getitem__] Called with idx={idx}")
        print(f"[__getitem__] model_type={self.model_type}")
        
        try:
            # Add detailed debug info
            print(f"[__getitem__] processed_data[{idx}] type: {type(self.processed_data[idx])}")
            print(f"[__getitem__] processed_data[{idx}] length: {len(self.processed_data[idx])}")
            print(f"[__getitem__] processed_data[{idx}] contents: {[type(x).__name__ for x in self.processed_data[idx]]}")
            
            indices, attention_mask, label = self.processed_data[idx]
            
            print(f"[__getitem__] Successfully unpacked: indices={type(indices).__name__}, attention_mask={type(attention_mask).__name__}, label={type(label).__name__}")
            
            # --- FIX: Conditionally return 2 or 3 items ---
            input_tensor = torch.tensor(indices, dtype=torch.long)
            label_tensor = torch.tensor([label], dtype=torch.long)

            if self.model_type == 'lstm':
                # LSTM needs input_ids and label (2 items)
                result = (input_tensor, label_tensor)
                print(f"[__getitem__] Returning 2 items for LSTM (indices, label)")
            else:
                # Transformer needs input_ids, attention_mask, and label (3 items)
                attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
                result = (input_tensor, attention_mask_tensor, label_tensor)
                print(f"[__getitem__] Returning 3 items for Transformer (indices, attention_mask, label)")
            # --- END FIX ---

            print(f"[__getitem__] Result shapes: {[x.shape for x in result]}")
            print(f"[__getitem__] Result types: {[type(x).__name__ for x in result]}")
            print(f"[__getitem__] About to return result with len={len(result)}")
            print(f"{'='*60}\n")
            return result
            
        except Exception as e:
            print(f"[__getitem__] ERROR: {type(e).__name__}: {e}")
            print(f"[__getitem__] processed_data[{idx}] = {self.processed_data[idx]}")
            import traceback
            traceback.print_exc()
            raise
        
# LSTM model
class LSTM(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=256, output_dim=2,
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
                 dim_feedforward=512, output_dim=2, dropout=0.1, pad_idx=0, max_len=512):
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
    
    print(f"\n[load_and_preprocess_data] DataLoader created for data_type={data_type}, model_type={model_type}")
    print(f"[load_and_preprocess_data] Testing first batch...")
    
    # Get first batch to see structure
    first_batch = next(iter(data_loader))
    print(f"[load_and_preprocess_data] Batch type: {type(first_batch)}")
    print(f"[load_and_preprocess_data] Batch is tuple/list with length: {len(first_batch)}")
    print(f"[load_and_preprocess_data] Batch item types: {[type(x).__name__ for x in first_batch]}")
    print(f"[load_and_preprocess_data] Batch item shapes: {[x.shape if hasattr(x, 'shape') else 'no shape' for x in first_batch]}")
    
    # DETAILED BATCH CONTENT EXAMINATION
    print(f"\n[BATCH CONTENTS DEBUG]")
    for i, item in enumerate(first_batch):
        print(f"\n  Batch element [{i}]:")
        print(f"    Type: {type(item)}")
        if hasattr(item, 'shape'):
            print(f"    Shape: {item.shape}")
            print(f"    Dtype: {item.dtype}")
            if item.numel() <= 20:
                print(f"    Full content: {item}")
            else:
                print(f"    First 10 values: {item.flatten()[:10].tolist()}")
                print(f"    Last 5 values: {item.flatten()[-5:].tolist()}")
        else:
            print(f"    Content: {item}")
    
    # Try to unpack as tests would
    print(f"\n[UNPACKING TEST]")
    try:
        if model_type == 'lstm':
            print(f"  Attempting LSTM unpacking: inputs, labels = batch")
            inputs, labels = first_batch
            print(f"  ✓ Success! inputs.shape={inputs.shape}, labels.shape={labels.shape}")
        else:
            print(f"  Attempting Transformer unpacking: inputs, attention_mask, labels = batch")
            inputs, attention_mask, labels = first_batch
            print(f"  ✓ Success! inputs.shape={inputs.shape}, attention_mask.shape={attention_mask.shape}, labels.shape={labels.shape}")
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"[load_and_preprocess_data] DONE\n")
    
    # Recreate dataloader since we consumed one batch
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
    
    for batch_idx, batch in enumerate(iterator):
        print(f"\n[train] === BATCH {batch_idx} ===")
        print(f"[train] Received batch with {len(batch)} items")
        print(f"[train] Batch type: {type(batch)}")
        print(f"[train] Batch item types: {[type(x).__name__ for x in batch]}")
        print(f"[train] Batch shapes: {[x.shape for x in batch]}")
        
        # Show first few values of each tensor
        print(f"[train] Batch contents preview:")
        for i, item in enumerate(batch):
            if hasattr(item, 'flatten'):
                print(f"[train]   Item {i}: first 5 values = {item.flatten()[:5].tolist()}")
        
        # --- FIX: Conditionally unpack 2 or 3 items ---
        if model_type == 'transformer':
            try:
                input_ids, attention_mask, labels = batch
                print(f"[train] ✓ Successfully unpacked 3 items for Transformer")
            except Exception as e:
                print(f"[train] ✗ FAILED to unpack 3 items: {type(e).__name__}: {e}")
                raise
            
            # Move all 3 to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            print(f"[train] Unpacked: input_ids.shape={input_ids.shape}, attention_mask.shape={attention_mask.shape}, labels.shape={labels.shape}")
        else:
            # For LSTM (2 items)
            try:
                input_ids, labels = batch
                print(f"[train] ✓ Successfully unpacked 2 items for LSTM")
            except Exception as e:
                print(f"[train] ✗ FAILED to unpack 2 items: {type(e).__name__}: {e}")
                raise
            
            attention_mask = None # Set explicitly for clarity, though unused
            
            # Move only 2 to device
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            print(f"[train] Unpacked: input_ids.shape={input_ids.shape}, labels.shape={labels.shape}") # Adjusted print statement
        # --- END FIX ---
        
        optimizer.zero_grad()
        
        # Use attention_mask only for transformer
        if model_type == 'transformer':
            print(f"[train] Calling transformer model with attention_mask")
            predictions = model(input_ids, attention_mask)
        else:
            print(f"[train] Calling LSTM model WITHOUT attention_mask")
            predictions = model(input_ids)
        
        loss = criterion(predictions, labels)
        acc = ((predictions.argmax(1) == labels).float().mean())
        
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
        for batch_idx, batch in enumerate(iterator):
            print(f"\n[evaluate] === BATCH {batch_idx} ===")
            print(f"[evaluate] Received batch with {len(batch)} items")
            print(f"[evaluate] Batch type: {type(batch)}")
            print(f"[evaluate] Batch item types: {[type(x).__name__ for x in batch]}")
            print(f"[evaluate] Batch shapes: {[x.shape for x in batch]}")
            
            # Show first few values of each tensor
            print(f"[evaluate] Batch contents preview:")
            for i, item in enumerate(batch):
                if hasattr(item, 'flatten'):
                    print(f"[evaluate]   Item {i}: first 5 values = {item.flatten()[:5].tolist()}")
            
            # --- FIX: Conditionally unpack 2 or 3 items ---
            if model_type == 'transformer':
                try:
                    input_ids, attention_mask, labels = batch
                    print(f"[evaluate] ✓ Successfully unpacked 3 items for Transformer")
                except Exception as e:
                    print(f"[evaluate] ✗ FAILED to unpack 3 items: {type(e).__name__}: {e}")
                    raise
                
                # Move all 3 to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                print(f"[evaluate] Unpacked: input_ids.shape={input_ids.shape}, attention_mask.shape={attention_mask.shape}, labels.shape={labels.shape}")
            else:
                # For LSTM (2 items)
                try:
                    input_ids, labels = batch
                    print(f"[evaluate] ✓ Successfully unpacked 2 items for LSTM")
                except Exception as e:
                    print(f"[evaluate] ✗ FAILED to unpack 2 items: {type(e).__name__}: {e}")
                    raise
                
                attention_mask = None # Set explicitly for clarity, though unused
                
                # Move only 2 to device
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                print(f"[evaluate] Unpacked: input_ids.shape={input_ids.shape}, labels.shape={labels.shape}") # Adjusted print statement
            # --- END FIX ---
            
            # Use attention_mask only for transformer
            if model_type == 'transformer':
                print(f"[evaluate] Calling transformer model with attention_mask")
                predictions = model(input_ids, attention_mask)
            else:
                print(f"[evaluate] Calling LSTM model WITHOUT attention_mask")
                predictions = model(input_ids)
            
            loss = criterion(predictions, labels)
            acc = ((predictions.argmax(1) == labels).float().mean())
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    EPOCHS = 10
    
    # Train LSTM
    MODEL_TYPE_LSTM = 'lstm'
    print(f"\n--- Training {MODEL_TYPE_LSTM.upper()} ---")
    
    train_loader_lstm, vocab_lstm = load_and_preprocess_data(
        data_path='train.parquet', 
        data_type='train', 
        model_type=MODEL_TYPE_LSTM
    )
    
    val_loader_lstm = load_and_preprocess_data(
        data_path='train.parquet', 
        data_type='val', 
        model_type=MODEL_TYPE_LSTM, 
        shared_vocab=vocab_lstm
    )
    
    print(f"LSTM Vocabulary size: {vocab_lstm.size}")
    print(f"LSTM Training samples: {len(train_loader_lstm.dataset)}")
    print(f"LSTM Validation samples: {len(val_loader_lstm.dataset)}")
    
    model_lstm = LSTM(
        vocab_size=vocab_lstm.size,
        embedding_dim=100,
        hidden_dim=256,
        output_dim=2,
        n_layers=2,
        bidirectional=True,
        dropout=0.5,
        pad_idx=vocab_lstm.word2idx['<pad>']
    ).to(device)
    
    criterion_lstm = nn.CrossEntropyLoss()
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.001)
    
    print(f"\nTraining {MODEL_TYPE_LSTM.upper()} model...")
    
    best_val_acc_lstm = 0.0
    for epoch in range(EPOCHS):
        print(f"\nLSTM Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train(model_lstm, train_loader_lstm, optimizer_lstm, criterion_lstm, device, MODEL_TYPE_LSTM)
        val_loss, val_acc = evaluate(model_lstm, val_loader_lstm, criterion_lstm, device, MODEL_TYPE_LSTM)
        
        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc_lstm:
            best_val_acc_lstm = val_acc
            torch.save(model_lstm.state_dict(), 'lstm.pt')
            print(f"Saved best LSTM model: {val_acc*100:.2f}%")
    
    print(f"\nBest LSTM validation accuracy: {best_val_acc_lstm*100:.2f}%")
    
    # Train Transformer
    MODEL_TYPE_TRANSFORMER = 'transformer'
    print(f"\n--- Training {MODEL_TYPE_TRANSFORMER.upper()} ---")
    
    train_loader_transformer, vocab_transformer = load_and_preprocess_data(
        data_path='train.parquet', 
        data_type='train', 
        model_type=MODEL_TYPE_TRANSFORMER
    )
    
    val_loader_transformer = load_and_preprocess_data(
        data_path='train.parquet', 
        data_type='val', 
        model_type=MODEL_TYPE_TRANSFORMER, 
        shared_vocab=vocab_transformer
    )
    
    print(f"Transformer Vocabulary size: {vocab_transformer.size}")
    print(f"Transformer Training samples: {len(train_loader_transformer.dataset)}")
    print(f"Transformer Validation samples: {len(val_loader_transformer.dataset)}")
    
    model_transformer = TransformerEncoder(
        vocab_size=vocab_transformer.size,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        output_dim=2,
        dropout=0.05,
        pad_idx=vocab_transformer.word2idx['<pad>'],
        max_len=512
    ).to(device)
    
    criterion_transformer = nn.CrossEntropyLoss()
    optimizer_transformer = optim.Adam(model_transformer.parameters(), lr=0.001)
    
    print(f"\nTraining {MODEL_TYPE_TRANSFORMER.upper()} model...")
    
    best_val_acc_transformer = 0.0
    for epoch in range(EPOCHS):
        print(f"\nTransformer Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train(model_transformer, train_loader_transformer, optimizer_transformer, criterion_transformer, device, MODEL_TYPE_TRANSFORMER)
        val_loss, val_acc = evaluate(model_transformer, val_loader_transformer, criterion_transformer, device, MODEL_TYPE_TRANSFORMER)
        
        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc_transformer:
            best_val_acc_transformer = val_acc
            torch.save(model_transformer.state_dict(), 'transformer.pt')
            print(f"Saved best Transformer model: {val_acc*100:.2f}%")
    
    print(f"\nBest Transformer validation accuracy: {best_val_acc_transformer*100:.2f}%")
    print("\n--- Training Complete ---")

if __name__ == "__main__":
    main()
