
#!/usr/bin/env python3
"""
DIRECT (NO-MCTS) GENERATION FROM FINE-TUNED CHEMGPT
"""
import json
import random
import re
import numpy as np
import pandas as pd
import selfies as sf
import torch
from rdkit import Chem
from transformers import AutoModelForCausalLM

ADAPTER_DIR = "finetuned_chemgpt"
VOCAB_FILE = "vocab_json_file"
MODEL_DIR = "merged_model_path"
OUTPUT_FILE = "output_file_name.csv"
N_GENERATED = 1000
SEED = 42
TEMPERATURE = 1.0
TOP_K = 80
TOP_P = 0.97
REPETITION_PENALTY = 1.15
MIN_NEW_TOKENS = 20
MAX_NEW_TOKENS = 150
SA_BIN = 4
QED_BIN = 7

class AcheTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"
        self.pad_token_id = vocab.get("[PAD]", 0)
        self.bos_token_id = vocab.get("[BOS]", 1)
        self.eos_token_id = vocab.get("[EOS]", 2)
        self.unk_token_id = vocab.get("[UNK]", 3)

    def __len__(self):
        return len(self.vocab)
    def __call__(self,text,truncation=True,max_length=384,padding=False,add_special_tokens=False,return_tensors=None):
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(20, len(text) - i),0,-1):
                candidate = text[i:i + length]
                if candidate in self.vocab:
                    tokens.append(candidate)
                    i += length
                    matched = True
                    break
            if not matched:
                if text[i] in self.vocab:
                    tokens.append(text[i])
                else:
                    tokens.append(self.unk_token)
                i += 1
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        input_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        return {"input_ids": input_ids,"attention_mask": [1] * len(input_ids)}

    def encode(self, text, *args, **kwargs):
        return self(text,*args,**kwargs,)["input_ids"]

    def decode(self, token_ids, skip_special_tokens=False):
        tokens = [self.ids_to_tokens.get(int(token_id),self.unk_token,)
            for token_id in token_ids]

        if skip_special_tokens:
            tokens = [token
                for token in tokens
                if token not in {self.pad_token,self.bos_token,self.eos_token,}]
        return "".join(tokens)

with open(VOCAB_FILE, "r") as f:
    vocab = json.load(f)
tokenizer = AcheTokenizer(vocab)
dtype = (torch.float16
    if torch.cuda.is_available()
    else torch.float32)

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR,torch_dtype=dtype,device_map="auto" if torch.cuda.is_available() else None,)
model.eval()

MODEL_DEVICE = next(model.parameters()).device
SA_TOKEN = f"<SA_BIN_{SA_BIN}>"
QED_TOKEN = f"<QED_BIN_{QED_BIN}>"

for required_token in [SA_TOKEN,QED_TOKEN,tokenizer.bos_token,]:
    if required_token not in vocab:
        raise KeyError(f"Required token {required_token} is absent from vocabulary.")
prompt = (f"{SA_TOKEN}"f"{QED_TOKEN}"f"{tokenizer.bos_token}")

def generate_one():
    input_ids = torch.tensor([tokenizer.encode(prompt)],dtype=torch.long,device=MODEL_DEVICE,)
    outputs = model.generate(input_ids=input_ids,do_sample=True,temperature=TEMPERATURE,top_k=TOP_K,top_p=TOP_P,repetition_penalty=REPETITION_PENALTY,
        min_new_tokens=MIN_NEW_TOKENS,max_new_tokens=MAX_NEW_TOKENS,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,)
    generated_ids = outputs[0].tolist()
    generated_part = generated_ids[input_ids.shape[1]:]
    raw_text = tokenizer.decode(generated_part,skip_special_tokens=False)
    raw_text = re.sub(r"<SA_BIN_\d+>", "", raw_text)
    raw_text = re.sub(r"<QED_BIN_\d+>", "", raw_text)
    raw_text = raw_text.replace("[PAD]", "")
    raw_text = raw_text.replace("[BOS]", "")
    raw_text = raw_text.replace("[EOS]", "")
    raw_text = raw_text.replace("[UNK]", "")
    selfies_string = "".join(re.findall(r"\[[^\]]+\]",raw_text,))
    if not selfies_string:
        return raw_text, None, None
    try:
        smiles = sf.decoder(selfies_string)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return selfies_string, smiles, None
        canonical = Chem.MolToSmiles(mol,canonical=True)
        return selfies_string, smiles, canonical
    except Exception:
        return selfies_string, None, None

for i in range(1, N_GENERATED + 1):
    selfies_string, raw_smiles, canonical_smiles = generate_one()
    records.append(
        {"attempt": i,"generator": "ChemGPT_direct","temperature": TEMPERATURE,"top_k": TOP_K,"top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,"SA_bin": SA_BIN,"QED_bin": QED_BIN,
            "selfies": selfies_string,"raw_smiles": raw_smiles,"canonical_smiles": canonical_smiles,"valid": canonical_smiles is not None,})
    if i % 100 == 0:
        print(f"Generated {i}/{N_GENERATED}")

df = pd.DataFrame(records)
df.to_csv(OUTPUT_FILE, index=False)
n_valid = int(df["valid"].sum())
n_unique_valid = (df.loc[df["valid"], "canonical_smiles"].nunique())
