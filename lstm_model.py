# PROPERTY-CONDITIONED LSTM TRAINING FOR SMILES GENERATION
import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

CSV_FILE = "name_of_input_file.csv"
SMILES_COLUMN = "SMILES"
SA_COLUMN = "SA_score"
QED_COLUMN = "QED"
OUTPUT_DIR = "lstm"
MODEL_FILE = os.path.join(OUTPUT_DIR, "lstm_model.pt")
VOCAB_FILE = os.path.join(OUTPUT_DIR, "lstm_vocab.json")
BIN_FILE = os.path.join(OUTPUT_DIR, "sa_qed_bin_edges.npz")
SEED = 42
BATCH_SIZE = 16
EPOCHS = 100
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.35
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
VALIDATION_FRACTION = 0.10
MAX_LENGTH = 150
N_PROPERTY_BINS = 10
GRAD_CLIP_NORM = 1.0
LOSS_IGNORE_INDEX = -100

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

def make_quantile_edges(values, n_bins=10):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No finite values available for binning.")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles).astype(float)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges

def assign_quantile_bin(value, edges):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Non-finite property value encountered.")
    bin_idx = int(np.digitize(value,edges[1:-1],right=False))
    return max(0,min(bin_idx, len(edges) - 2))

class SMILESTokenizer:
    def __init__(self, n_property_bins=10):
        self.n_property_bins = n_property_bins
        self.base_special_tokens = ["[PAD]","[BOS]","[EOS]","[UNK]"]
        self.sa_tokens = [f"<SA_BIN_{i}>"for i in range(n_property_bins)]
        self.qed_tokens = [f"<QED_BIN_{i}>"for i in range(n_property_bins)]
        self.special_tokens = (self.base_special_tokens+ self.sa_tokens+ self.qed_tokens)
        self.vocab = None
        self.token_to_id = None
        self.id_to_token = None

    def tokenize(self, smiles):

        pattern = (r"(\[[^\]]+\]"r"|Br|Cl"r"|Si|Se|Na|Li|Al|Ca|Fe|Mg|Zn"r"|[BCNOFPSKI]"r"|[bcnops]"r"|[0-9]"r"|"r"[-=#:.~/\\]"r"|"r"[()]"r"|"r"[+*@?]"r"|"r"[H]"r")")
        tokens = re.findall(pattern, smiles)
        reconstructed = "".join(tokens)
        if reconstructed != smiles:
            return None
        return tokens

    def build_vocab(self, smiles_list):
        token_set = set()
        valid_count = 0
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            if tokens is None:
                continue
            valid_count += 1
            for token in tokens:
                token_set.add(token)
        molecular_tokens = sorted(token_set)
        self.vocab = (self.special_tokens+ molecular_tokens)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        return self.vocab

    def encode_conditioned(self,smiles,sa_bin,qed_bin,max_length=None):
        tokens = self.tokenize(smiles)
        if tokens is None:
            return None
        sa_token = f"<SA_BIN_{int(sa_bin)}>"
        qed_token = f"<QED_BIN_{int(qed_bin)}>"
        if sa_token not in self.token_to_id:
            raise KeyError(f"Missing conditioning token: {sa_token}")
        if qed_token not in self.token_to_id:
            raise KeyError(f"Missing conditioning token: {qed_token}")
        ids = [self.token_to_id[sa_token],self.token_to_id[qed_token],self.token_to_id["[BOS]"]]

        for token in tokens:
            ids.append(self.token_to_id.get(token,self.token_to_id["[UNK]"]))
        ids.append(self.token_to_id["[EOS]"])

        if max_length is not None:
            if len(ids) > max_length:
                return None
            pad_id = self.token_to_id["[PAD]"]
            ids += [pad_id] * (max_length - len(ids))
        return ids

    def decode(self, ids):
        tokens = []
        skip_tokens = set(self.special_tokens)

        for idx in ids:
            token = self.id_to_token[int(idx)]
            if token == "[EOS]":
                break
            if token in skip_tokens:
                continue
            tokens.append(token)
        return "".join(tokens)

    def save(self, path):
        data = {"vocab": self.vocab,
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "n_property_bins": self.n_property_bins,
            "conditioning_format":"<SA_BIN_i> <QED_BIN_j> [BOS] SMILES [EOS]"}

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        tokenizer = cls(n_property_bins=data.get("n_property_bins",10))
        tokenizer.vocab = data["vocab"]
        tokenizer.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tokenizer.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        return tokenizer

df = pd.read_csv(CSV_FILE)
required_columns = [SMILES_COLUMN, SA_COLUMN, QED_COLUM]
df = df.dropna(subset=required_columns).copy()
df[SMILES_COLUMN] = (df[SMILES_COLUMN].astype(str).str.strip())
df[SA_COLUMN] = pd.to_numeric(df[SA_COLUMN])
df[QED_COLUMN] = pd.to_numeric(df[QED_COLUMN])
df = df.dropna(subset=[SA_COLUMN, QED_COLUMN]).copy()

canonical_smiles = []
for smiles in df[SMILES_COLUMN]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        canonical_smiles.append(None)
    else:
        canonical_smiles.append(Chem.MolToSmiles(mol,canonical=True))
df["Canonical_SMILES"] = canonical_smiles
invalid_count = (df["Canonical_SMILES"].isna().sum())

df = (df.dropna(subset=["Canonical_SMILES"]).copy()
before_duplicates = len(df)

rng = np.random.default_rng(SEED)
indices = np.arange(len(df))
rng.shuffle(indices)

val_size = max(1,int(len(indices) * VALIDATION_FRACTION))
val_indices = indices[:val_size]
train_indices = indices[val_size:]
train_df = (df.iloc[train_indices].reset_index(drop=True))
val_df = (df.iloc[val_indices].reset_index(drop=True))

sa_edges = make_quantile_edges(train_df[SA_COLUMN].values,n_bins=N_PROPERTY_BINS)
qed_edges = make_quantile_edges(train_df[QED_COLUMN].values,n_bins=N_PROPERTY_BINS)
train_df["SA_bin"] = (train_df[SA_COLUMN].apply(lambda x: assign_quantile_bin(x,sa_edges)))
train_df["QED_bin"] = (train_df[QED_COLUMN].apply(lambda x: assign_quantile_bin(x,qed_edges)))
val_df["SA_bin"] = (val_df[SA_COLUMN].apply(lambda x: assign_quantile_bin(x,sa_edges)))
val_df["QED_bin"] = (val_df[QED_COLUMN].apply(lambda x: assign_quantile_bin(x,qed_edges)))

tokenizer = SMILESTokenizer(n_property_bins=N_PROPERTY_BINS)
vocab = tokenizer.build_vocab(train_df["Canonical_SMILES"].tolist())
vocab_size = len(vocab)

def encode_dataframe(data,tokenizer,max_length,dataset_name):
    encoded_sequences = []
    skipped = 0
    for _, row in data.iterrows():
        encoded = tokenizer.encode_conditioned(smiles=row["Canonical_SMILES"],sa_bin=row["SA_bin"],qed_bin=row["QED_bin"],max_length=max_length)
        if encoded is None:
            skipped += 1
            continue
        encoded_sequences.append(encoded)
    return encoded_sequences

encoded_train = encode_dataframe(train_df,tokenizer,MAX_LENGTH,"Training")
encoded_val = encode_dataframe(val_df,tokenizer,MAX_LENGTH,"Validation")

class ConditionedSMILESDataset(Dataset):
    def __init__(self,sequences,pad_id):
        self.data = torch.tensor(sequences,dtype=torch.long)
        self.pad_id = pad_id
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sequence = self.data[idx]
        x = sequence[:-1].clone()
        y = sequence[1:].clone()
        y[:2] = LOSS_IGNORE_INDEX
        y[y == self.pad_id] = LOSS_IGNORE_INDEX
        return x, y
pad_id = tokenizer.token_to_id["[PAD]"]
train_dataset = ConditionedSMILESDataset(encoded_train,pad_id)
val_dataset = ConditionedSMILESDataset(encoded_val,pad_id)

class SMILESLSTM(nn.Module):
    def __init__(self,vocab_size,pad_id,embed_dim=128,hidden_dim=256,num_layers=2,dropout=0.35):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim,padding_idx=pad_id)
        self.lstm = nn.LSTM(input_size=embed_dim,hidden_size=hidden_dim,num_layers=num_layers,batch_first=True,dropout=(dropout if num_layers > 1else 0.0))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim,vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits
model = SMILESLSTM(vocab_size=vocab_size,pad_id=pad_id,embed_dim=EMBED_DIM,hidden_dim=HIDDEN_DIM,num_layers=NUM_LAYERS,dropout=DROPOUT).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=LOSS_IGNORE_INDEX)
optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=3,min_lr=1e-6)

def train_one_epoch(model,loader,optimizer,criterion):
    model.train()
    for x, y in loader:
        x = x.to(non_blocking=True)
        y = y.to(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.reshape(-1,logits.size(-1)),y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return (total_loss /max(total_batches, 1))

def validate(model,loader,criterion):
    model.eval()
    for x, y in loader:
        x = x.to(non_blocking=True)
        y = y.to(non_blocking=True)
        logits = model(x)
        loss = criterion(logits.reshape(-1,logits.size(-1)),y.reshape(-1))
        total_loss += loss.item()
        total_batches += 1
    return (total_loss /max(total_batches, 1))

best_val_loss = float("inf")
history = []
patience = 10
epochs_without_improvement = 0

for epoch in range(1,EPOCHS + 1):
    train_loss = train_one_epoch(model,optimizer,criterion)
    val_loss = validate(model, criterion)
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    history.append({"epoch": epoch,"train_loss": train_loss,"val_loss": val_loss,"learning_rate": current_lr})
    min_delta = 1e-4
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        checkpoint = {"model_state_dict": model.state_dict(),"vocab_size": vocab_size,"embed_dim": EMBED_DIM,"hidden_dim": HIDDEN_DIM,"num_layers": NUM_LAYERS,
            "dropout": DROPOUT,"max_length": MAX_LENGTH,"best_val_loss": best_val_loss,"n_property_bins": N_PROPERTY_BINS,
            "sa_edges": sa_edges,"qed_edges": qed_edges,"conditioning_format":"<SA_BIN_i> <QED_BIN_j> [BOS] SMILES [EOS]",
            "pad_id": tokenizer.token_to_id["[PAD]"],"bos_id": tokenizer.token_to_id["[BOS]"],"eos_id": tokenizer.token_to_id["[EOS]"]}
        torch.save(checkpoint,MODEL_FILE)
    else:
        epochs_without_improvement += 1
    if (epochs_without_improvement >= patience):
        print("Early stopping triggered.")
        break