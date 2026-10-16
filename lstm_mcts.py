# LSTM-MCTS MOLECULE GENERATION PIPELINE
import os
import re
import json
import time
import random
import subprocess
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import (Descriptors,Lipinski,QED,rdMolDescriptors,FilterCatalog)
from rdkit.Chem import AllChem
from rdkit import DataStructs
from training_advanced import EdgeAwareGCNNet
from Smiles_to_graph import smiles_to_graph
from torch.utils.data import Dataset, DataLoader


CSV_FILE = "_input_file_name.csv"
OUTPUT_DIR = "output_folde_name"
SMILES_COLUMN = "SMILES"
SA_COLUMN = "SA_score"
QED_COLUMN = "QED"
SEED = 42
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.35
BATCH_SIZE = 50 #128
EPOCHS = 30
LEARNING_RATE = 1e-3
MAX_LENGTH = 150
N_GENERATED = 1000
TEMPERATURE = 0.8
TOP_K = 80
TOP_P = 0.95
MIN_NEW_TOKENS = 10
MAX_NEW_TOKENS = 150
BATCH_DOCK_SIZE = 50
MCTS_ITERATIONS = N_GENERATED
MCTS_EXPANSION_TOP_K = 10
MCTS_MAX_DEPTH = MAX_NEW_TOKENS
UCT_C = 2.0
MCTS_ROLLOUT_TRIES = 1
MCTS_REJECTION_REWARD = -0.10
W_GNN = 0.40
W_DOCK = 0.30
W_QED = 0.10
W_SA = 0.10
W_NOVELTY = 0.10
QED_MIN = 0.40
SA_MAX = 7.0
GNN_THRESHOLD = 0.60
LSTM_THRESHOLD = 0.60

RDOCK_BIN = ("path_of_rbdock")
RECEPTOR = "dock_receptor.prm"
PROTOCOL = "dock.prm"
RDOCK_N_POSES = 20

df = pd.read_csv(CSV_FILE)
if SMILES_COLUMN not in df.columns:
    raise ValueError(f"SMILES column '{SMILES_COLUMN}' not found")

smiles_list = (df[SMILES_COLUMN].dropna().astype(str).tolist())

class SMILESTokenizer:
    def __init__(self):
        self.special_tokens = ["[PAD]","[BOS]","[EOS]","[UNK]"]
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
        tokens = sorted(token_set)
        self.vocab = (self.special_tokens+tokens)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        return self.vocab

    @classmethod
    def load(cls, filepath):
        tokenizer = cls()
        with open(filepath, "r") as f:
            data = json.load(f)
        if "vocab" not in data:
            raise KeyError("vocab not found")
        tokenizer.vocab = data["vocab"]
        if "token_to_id" in data:
            tokenizer.token_to_id = {str(token): int(idx) for token, idx in data["token_to_id"].items()}
        else:
            tokenizer.token_to_id = {token: i for i, token in enumerate(tokenizer.vocab)}
        if "id_to_token" in data:
            tokenizer.id_to_token = {int(idx): token for idx, token in data["id_to_token"].items()}
        else:
            tokenizer.id_to_token = {i: token for i, token in enumerate(tokenizer.vocab)}
        if len(tokenizer.vocab) != len(tokenizer.token_to_id):
            raise ValueError("Vocabulary and token id size mismatch.")
        if len(tokenizer.vocab) != len(tokenizer.id_to_token):
            raise ValueError("Vocabulary and idtoken size mismatch.")

        required_tokens = ["[PAD]","[BOS]","[EOS]","[UNK]"]
        for token in required_tokens:
            if token not in tokenizer.token_to_id:
                raise ValueError(f"Required token '{token}' "f"not found in {filepath}")
        return tokenizer

def top_k_top_p_filter(logits,top_k=0,top_p=1.0):
    logits = logits.clone()
    if top_k > 0:
        top_k = min(top_k,logits.size(-1))
        threshold = torch.topk(logits,top_k).values[-1]
        logits[logits < threshold] = -float("inf")
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits,descending=True)
        probabilities = torch.softmax(sorted_logits,dim=-1)
        cumulative_probs = torch.cumsum(probabilities,dim=-1)
        remove = cumulative_probs > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        indices_to_remove = (sorted_indices[remove])
        logits[indices_to_remove] = -float("inf")
    return logits

def generate_smiles(model,temperature=1.0,top_k=80,top_p=0.95,max_length=150):
    model.eval()
    generated = [BOS_ID]
    log_probabilities = []
    with torch.no_grad():
        for step in range(max_length):
            x = torch.tensor([generated],dtype=torch.long,device=DEVICE)
            logits = model(x)
            next_logits = logits[0, -1, :].clone()
            next_logits = next_logits / max(temperature, 1e-8)
            next_logits[PAD_ID] = -float("inf")
            if step < MIN_NEW_TOKENS:
                next_logits[EOS_ID] = -float("inf")
            filtered = top_k_top_p_filter(next_logits,top_k=top_k,top_p=top_p)
            probabilities = torch.softmax(filtered,dim=-1)
            
            if not torch.isfinite(probabilities).all():
                break
            next_token = torch.multinomialprobabilities,num_samples=1).item()
            token_prob = probabilities[next_token].item()
            log_probabilities.append(np.log(max(token_prob, 1e-12)))
            generated.append(next_token)
            if next_token == EOS_ID:
        tokens = []

    for idx in generated:
        if idx in ( BOS_ID,EOS_ID,PAD_ID):
            continue
        tokens.append(itos.get(idx, ""))
    smiles = "".join(tokens)
    if len(log_probabilities) > 0:
        mean_log_prob = np.mean(log_probabilities)
       confidence = float(np.exp(mean_log_prob))
    else:
        confidence = 0.0
    return smiles, confidence

def lstm_next_token_probabilities(model,token_sequence,temperature=1.0,top_k=80,top_p=0.95):
    model.eval()
    with torch.no_grad():x = torch.tensor(token_sequence,dtype=torch.long,device=DEVICE)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = model(x)
        next_logits = logits[0, -1, :].clone()
        next_logits = next_logits / max(temperature, 1e-8)
        next_logits[PAD_ID] = -float("inf")
        filtered = top_k_top_p_filter(next_logits,top_k=top_k,top_p=top_p)
        probabilities = torch.softmax(filtered,dim=-1)
        if not torch.isfinite(probabilities).all():
            return None
        return probabilities.cpu().numpy()

def relaxed_ring_check(mol):
    try:
        ri = mol.GetRingInfo()
        if not ri.NumRings():
            return True
        rings = ri.AtomRings()
        ring_sizes = [len(set(r))for r in rings]
        if any(size < 3 or size > 7 for size in ring_sizes):
            return False
        if len(rings) > 5:
            return False
        fused_pairs = 0
        for i, r1 in enumerate(rings):
            for r2 in rings[i + 1:]:
                shared_atoms = len(set(r1) &set(r2))
                if shared_atoms >= 2:
                    fused_pairs += 1
        if fused_pairs > 1:
            return False
        return True
    except Exception:
        return False

def pains_filter(mol):
    try:
        matches = pains_catalog.GetMatches(mol)
        return len(matches) == 0
    except Exception:
        return False

def synthetic_accessibility(mol):
    heavy = mol.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    score = (2.5 +0.04 * heavy +0.35 * rings +0.08 * rot)
    return float(np.clip(score, 1.0, 10.0))

def get_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048)

def calculate_novelty(mol, train_fps):
    fp = get_fingerprint(mol)
    if len(train_fps) == 0:
        return 1.0
    similarities = [DataStructs.TanimotoSimilarity(fp, x)for x in train_fps]
    max_sim = max(similarities)
    return 1.0 - max_sim

def run_batch_docking(sdf_path,out_prefix="batch_out"):
    cmd = [RDOCK_BIN,"-r",RECEPTOR,"-p",PROTOCOL,"-i",sdf_path,"-o",out_prefix,"-n",str(RDOCK_N_POSES)]
    try:
        result = subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        print(result.stdout[-1000:])
    except subprocess.CalledProcessError as e:
        print("rDock failed:")
        print(e.stderr)
        return None
    candidates = [out_prefix + ".sd",out_prefix + ".sdf",out_prefix]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def canonicalize_smiles(smiles):
    try:
        if smiles is None:
            return None
        smiles = str(smiles).strip()
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None

def parse_docking_scores(sd_file,batch_smiles):
    input_canonical = {}
    for smi in batch_smiles:
        csmi = canonicalize_smiles(smi)
        if csmi is not None:
            input_canonical[csmi] = smi
    docking_scores = defaultdict(list)
    with open(sd_file,"r",errors="ignore") as f:
        text = f.read()
    records = text.split("$$")
    pose_records = 0
    score_records = 0
    matched_records = 0
    for record in records:
        if "<SCORE>" not in record:
           continue
        pose_records += 1
        score_match = re.search(r">\<SCORE>\n"
            r"\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)",record)
        if score_match is None:
            continue
        score_records += 1
        try:
            score = float(score_match.group(1))
        except Exception:
            continue

        smiles_match = re.search(r">\<SMILES>\n"r"(.+)",record)
        if smiles_match is None:
            continue
        dock_smi = (smiles_match.group(1).strip())
        canonical_dock = (canonicalize_smiles(dock_smi))
        if canonical_dock in input_canonical:
            original = input_canonical[canonical_dock]
            docking_scores[original].append(score)
            matched_records += 1
    for smi, scores in docking_scores.items():
        print(f"{smi} -> "f"{len(scores)} poses | "f"best SCORE = "f"{min(scores):.4f}")
    return dict(docking_scores)

class Node:
    def __init__(self,token_sequence,parent=None,token_id=None):
        self.token_sequence = list(token_sequence)
        self.parentNode = parent
        self.token_id = token_id
        self.childNodes = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = None
        self.is_terminal = False

    def Addnode(self,token_id):
        child_sequence = (self.token_sequence + [token_id])
        child = Node(token_sequence=child_sequence,parent=self,token_id=token_id)
        self.childNodes.append(child)
        return child

    def mean_reward(self):
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def UCT(self):
        if self.visits == 0:
            return float("inf")
        if self.parentNode is None:
            return self.mean_reward()

        exploitation = ( self.total_reward / self.visits)
        exploration = ( UCT_C *np.sqrt(np.log(self.parentNode.visits + 1.0) / self.visits))
        return exploitation + exploration

    def Selectnode(self):
        if not self.childNodes:
            return None

        return max(self.childNodes,key=lambda child: child.UCT())

    def Update(self, reward):
        self.visits += 1
        self.total_reward += float(reward)

def decode_mcts_tokens(token_sequence):
    tokens = []
    for idx in token_sequence:
        if idx in (BOS_ID,EOS_ID,PAD_ID):
            continue

        tokens.append(itos.get(idx, ""))
    return "".join(tokens)

def initialize_node_actions(node,model):
    if node.untried_actions is not None:
        return
    probabilities = lstm_next_token_probabilities(model=model,token_sequence=node.token_sequence,
        temperature=TEMPERATURE,top_k=TOP_K,top_p=TOP_P)
    
    if probabilities is None:
        node.untried_actions = []
        node.is_terminal = True
        return
    probabilities[PAD_ID] = 0.0
    generated_length = (len(node.token_sequence) - 1)

    if generated_length < MIN_NEW_TOKENS:
        probabilities[EOS_ID] = 0.0

    candidate_ids = np.argsort(probabilities)[::-1]
    for token_id in candidate_ids:
        probability = probabilities[token_id]
        if probability <= 0:
            continue
        actions.append((int(token_id),float(probability)))
        if len(actions) >= MCTS_EXPANSION_TOP_K:
            break
    node.untried_actions = actions
    if EOS_ID in [token_id for token_id, _ in actions]:
        pass

def expand_node(node,model):
    initialize_node_actions(node,model)
    if not node.untried_actions:
        return None
    token_id, probability = (node.untried_actions.pop(0))
    child = node.Addnode(token_id)
    child.lstm_probability = probability
    if token_id == EOS_ID:
        child.is_terminal = True
    elif len(child.token_sequence) >= MCTS_MAX_DEPTH + 1:
        child.is_terminal = True
    return child

def rollout_from_node(node,model):
    sequence = list(node.token_sequence)
    log_probabilities = []
    for step in range(MCTS_MAX_DEPTH):
        if sequence[-1] == EOS_ID:
            break
        probabilities = (lstm_next_token_probabilities(model=model,token_sequence=sequence,
                temperature=TEMPERATURE,top_k=TOP_K,top_p=TOP_P))
        if probabilities is None:
            break
        probabilities[PAD_ID] = 0.0
        generated_length = (len(sequence) - 1)
        if generated_length < MIN_NEW_TOKENS:
            probabilities[EOS_ID] = 0.0
        probability_sum = probabilities.sum()
        if probability_sum <= 0:
            break
        probabilities = (probabilities / probability_sum)

        token_id = int(np.random.choice(len(probabilities),p=probabilities))
        token_probability = (probabilities[token_id])
        log_probabilities.append(np.log(max(token_probability,1e-12)))
        sequence.append(token_id)
        if token_id == EOS_ID:
            break
    smiles = decode_mcts_tokens(sequence)
    if log_probabilities:
        confidence = float(np.exp(np.mean(log_probabilities)))
    else:
        confidence = 0.0
    return (smiles,confidence,sequence)

def select_mcts_node(root,model):
    node = root
    while True:
        initialize_node_actions(node,model)
        if node.untried_actions:
            return node
        if node.is_terminal:
            return node
        if not node.childNodes:
            return node
        node = node.Selectnode()
        if node is None:
            return None

def backpropagate_mcts(node,reward):
    current = node
    while current is not None:
        current.Update(reward)
        current = current.parentNode
 
def predict_gnn_inhibitor(smiles, gnn_model, feature_mean, feature_std):
    mol = Chem.MolFromSmiles(smiles)
    graph = smiles_to_graph(smiles, 0.0)
    device = next(gnn_model.parameters()).device
    graph = graph.to(device)
    feature_mean = torch.as_tensor(feature_mean,dtype=graph.x.dtype,device=device)
    feature_std = torch.as_tensor(feature_std,dtype=graph.x.dtype,device=device)
    feature_std = torch.where(feature_std == 0,torch.ones_like(feature_std),feature_std)
    graph.x = (graph.x - feature_mean) / feature_std
    batch = torch.zeros(graph.x.size(0),dtype=torch.long,device=device)
    gnn_model.eval()
    with torch.no_grad():
        out, _ = gnn_model(graph.x,graph.edge_index,graph.edge_attr,batch)
        probability = torch.sigmoid(out).view(-1)[0].item()
    return float(probability)

def calculate_reward(inhibitor_probability, dock, qed, sa, novelty):
    sa_normalized = np.clip((SA_MAX - sa) / SA_MAX,0.0,
    gnn_component = W_GNN * inhibitor_probability
    dock_component = W_DOCK * dock_normalized
    qed_component = W_QED * qed
    sa_component = W_SA * sa_normalized
    novelty_component = W_NOVELTY * novelty
    reward = (gnn_component+ dock_component+ qed_component+ sa_component+ novelty_component)

    return {"reward": reward,"dock_normalized": dock_normalized,"gnn_component": gnn_component,"dock_component": dock_component,
        "qed_component": qed_component,"sa_component": sa_component,"novelty_component": novelty_component}


def run_lstm_pipeline(model,train_smis,gnn_model,feature_mean,feature_std,morgan_gen):
    valid_compounds = [] 
    canonical_smiles = set()
    passed_smiles = []
    accepted_smiles = []
    generated_smiles_all = []
    batch_smiles = []
    batch_confidence = {}
    batch_gnn_probability = {}
    batch_iteration = {}
    batch_mcts_nodes = {}
 
    for s in train_smis:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            canonical_smiles.add(Chem.MolToSmiles(mol,canonical=True))
    seen_smiles = set(canonical_smiles)
    train_fps = []
    for smi in canonical_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            train_fps.append(get_fingerprint(mol))
    stats = {"generated": 0,"valid": 0,"ring_pass": 0,"druglike": 0,"non_pains": 0,"size_pass": 0,
        "gnn_pass": 0,"docked": 0,"qed_pass": 0,"sa_pass": 0,"accepted": 0}
    
    best_reward_per_iteration = []
    best_reward_so_far = -float("inf")
    root = Node(token_sequence=[BOS_ID], parent=None)   
    mcts_stats = {"selection": 0,"expansion": 0,"rollout": 0,"backpropagation": 0,}
   
    for iteration in range(1, MCTS_ITERATIONS + 1):
        try:
            selected_node = select_mcts_node(root, model)
            mcts_stats["selection"] += 1         
            if selected_node is None:
                continue         
            expanded_node = expand_node(selected_node, model)
            mcts_stats["expansion"] += 1                       
            if expanded_node is None:
                continue
            smi, confidence, final_sequence = rollout_from_node(expanded_node, model)
            mcts_stats["rollout"] += 1
            mcts_reward = MCTS_REJECTION_REWARD           
            stats["generated"]+=1          
            generated_smiles_all.append(smi)
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mcts_reward = MCTS_REJECTION_REWARD
                backpropagate_mcts(expanded_node, mcts_reward)
                mcts_stats["backpropagation"] += 1
                continue

            stats["valid"] += 1
            smi = Chem.MolToSmiles(mol,canonical=True)
            if smi in seen_smiles:
                mcts_reward = MCTS_REJECTION_REWARD
                backpropagate_mcts(expanded_node, mcts_reward)
                mcts_stats["backpropagation"] += 1
                continue
            seen_smiles.add(smi)

            if not relaxed_ring_check(mol):
                mcts_reward = MCTS_REJECTION_REWARD
                backpropagate_mcts(expanded_node, mcts_reward)
                mcts_stats["backpropagation"] += 1
                continue
            stats["ring_pass"] += 1
            
            if not strict_drug_like_filter(mol):
                mcts_reward = MCTS_REJECTION_REWARD
                backpropagate_mcts(expanded_node, mcts_reward)
                mcts_stats["backpropagation"] += 1
                continue          
            stats["druglike"] += 1

            if not pains_filter(mol):                
                mcts_reward = MCTS_REJECTION_REWARD
                backpropagate_mcts(expanded_node, mcts_reward)
                mcts_stats["backpropagation"] += 1
                continue
            stats["non_pains"] += 1

            heavy = mol.GetNumHeavyAtoms()
            rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if (heavy < 12 or heavy > 70 or rot > 12):
                print("Rejected: size/rotatable bond filter")
                mcts_reward = MCTS_REJECTION_REWARD
                backpropagate_mcts(expanded_node,mcts_reward)
                mcts_stats["backpropagation"] += 1
                continue
            stats["size_pass"] += 1

            inhibitor_probability = predict_gnn_inhibitor(smi,gnn_model,feature_mean,feature_std)
            if inhibitor_probability < GNN_THRESHOLD:
                mcts_reward = MCTS_REJECTION_REWARD
                backpropagate_mcts(expanded_node, mcts_reward)
                mcts_stats["backpropagation"] += 1
                continue
            stats["gnn_pass"] += 1
            
            batch_smiles.append(smi)
            batch_confidence[smi] = confidence
            batch_gnn_probability[smi] = inhibitor_probability
            batch_iteration[smi] = iteration
            batch_mcts_nodes[smi] = expanded_node
            if len(batch_smiles) < BATCH_DOCK_SIZE:
                continue
                
            sdf_path = smiles_list_to_sdf(batch_smiles)
                for smin in batch_smiles:
                    mcts_node =batch_mcts_nodes.get(smin)
                    if mcts_node is not None:
                        backpropagate_mcts(mcts_node, MCTS_REJECTION_REWARD)
                continue

            out_path = run_batch_docking(sdf_path)

            if (out_path is None or not os.path.exists(out_path)):
                for smin in batch_smiles:
                    mcts_node =batch_mcts_nodes.get(smin)
                    if mcts_node is not None:
                        backpropagate_mcts(mcts_node, MCTS_REJECTION_REWARD)
                        mcts_stats["backpropagation"]+=1
                continue
            scores = parse_docking_scores(out_path,batch_smiles)
            docked_count = 0
            
            for smin in batch_smiles:
                molecule_scores = scores.get(smin)
                if molecule_scores is None:
                    mcts_node = batch_mcts_nodes.get(smin)
                    if mcts_node is not None:
                        backpropagate_mcts(mcts_node,MCTS_REJECTION_REWARD)
                    mcts_stats["backpropagation"] += 1
                    continue
                if len(molecule_scores) == 0:
                    continue
                dock = min(molecule_scores)
                docked_count += 1
                
                mol = Chem.MolFromSmiles(smin)

                qed = QED.qed(mol)
                if qed < QED_MIN:
                    mcts_node =batch_mcts_nodes.get(smin)
                    if mcts_node is not None:
                        mcts_reward = MCTS_REJECTION_REWARD
                        backpropagate_mcts(mcts_node, mcts_reward)
                        mcts_stats["backpropagation"] += 1
                    continue
                    
                sa = synthetic_accessibility(mol)
                if sa > SA_MAX:
                    mcts_node = batch_mcts_nodes.get(smin)
                    if mcts_node is not None:
                        mcts_reward = MCTS_REJECTION_REWARD
                        backpropagate_mcts(mcts_node, mcts_reward)
                        mcts_stats["backpropagation"] += 1
                    continue

                novelty = calculate_novelty(mol, train_fps)
                lstm_confidence = batch_confidence.get(smin,0.0)
                inhibitor_probability = (batch_gnn_probability.get(smin,0.0))
                reward_data = calculate_reward(inhibitor_probability,dock,qed,sa,novelty)
                reward = reward_data["reward"]
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)
                tpsa = Descriptors.TPSA(mol)
                ring_count = (mol.GetRingInfo().NumRings())
                rot_bonds = (rdMolDescriptors.CalcNumRotatableBonds(mol))
                props = {"smiles":smin,"MW":mw,"LogP":logp,"HBD":hbd,"HBA":hba,"TPSA":tpsa,"RingCount":ring_count,"RotBonds":rot_bonds,"qed":qed,"sa_score":sa,
                    "lstm_confidence":lstm_confidence,"inhibitor_probability":inhibitor_probability,"dock_score":dock,"dock_normalized":reward_data["dock_normalized"],
                    "novelty":novelty,"gnn_component":reward_data["gnn_component"],"dock_component":reward_data["dock_component"],"qed_component":reward_data["qed_component"],
                    "sa_component":reward_data["sa_component" ],"novelty_component":reward_data["novelty_component"],"reward":reward,"iteration":batch_iteration.get(smin,iteration)}

                valid_compounds.append(props)
                accepted_smiles.append(smin)

                stats["accepted"] += 1
                passed_smiles.append(smin)
            stats["docked"] += docked_count
        finally:
            best_reward_per_iteration.append(best_reward_so_far if np.isfinite(best_reward_so_far) else np.nan)

    if len(batch_smiles) > 0:
        sdf_path = smiles_list_to_sdf(batch_smiles)
        if sdf_path is not None:
            out_path = run_batch_docking(sdf_path)
            if (out_path is not None and os.path.exists(out_path)):
                scores = parse_docking_scores(out_path,batch_smiles)
                for smin in batch_smiles:
                    molecule_scores = scores.get(smin)
                    if not molecule_scores:
                        mcts_node =batch_mcts_nodes.get(smin)
                        if mcts_node is not None:
                            backpropagate_mcts(mcts_node, MCTS_REJECTION_REWARD)
                            mcts_stats["backpropagation"]+=1
                        continue

                    dock = min(molecule_scores)
                    docked_count += 1
                    mol = Chem.MolFromSmiles(smin)

                    qed = QED.qed(mol)
                    if qed < QED_MIN:
                        mcts_node = batch_mcts_nodes.get(smin)
                        if mcts_node is not None:
                            backpropagate_mcts(mcts_node,MCTS_REJECTION_REWARD)
                            mcts_stats["backpropagation"] += 1
                        continue

                    sa = synthetic_accessibility(mol)
                    if sa > SA_MAX:
                        mcts_node = batch_mcts_nodes.get(smin)
                        if mcts_node is not None:
                            backpropagate_mcts(mcts_node,MCTS_REJECTION_REWARD)
                            mcts_stats["backpropagation"] += 1
                        continue

                    novelty = calculate_novelty(mol, train_fps)
                    lstm_confidence = (batch_confidence.get(smin,0.0))
                    inhibitor_probability = (batch_gnn_probability.get(smin,0.0))
                    reward_data = calculate_reward(inhibitor_probability,dock,qed,sa,novelty)
                    reward = reward_data["reward"]
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    hbd = Lipinski.NumHDonors(mol)
                    hba = Lipinski.NumHAcceptors(mol)
                    tpsa = Descriptors.TPSA(mol)
                    ring_count = (mol.GetRingInfo().NumRings())
                    rot_bonds = (rdMolDescriptors.CalcNumRotatableBonds(mol))

                    props = {"smiles":smin,"MW":mw,"LogP":logp,"HBD":hbd,"HBA":hba,"TPSA":tpsa,"RingCount":ring_count,"RotBonds":rot_bonds,"qed":qed,"sa_score":sa,
                    "lstm_confidence":lstm_confidence,"inhibitor_probability":inhibitor_probability,"dock_score":dock,"dock_normalized":reward_data["dock_normalized"],"novelty":novelty,
                    "gnn_component":reward_data["gnn_component"],"dock_component":reward_data["dock_component"],"qed_component":reward_data["qed_component"],
                    "sa_component":reward_data["sa_component"],"novelty_component":reward_data["novelty_component"],"reward":reward,"iteration":batch_iteration.get(smin,MCTS_ITERATIONS)}

                    if reward > best_reward_so_far:
                        best_reward_so_far = reward
                    best_reward_per_iteration.append(best_reward_so_far)                  
                    mcts_node = batch_mcts_nodes.get(smin)
                    
                    if mcts_node is not None:
                        backpropagate_mcts(mcts_node,reward)
                        mcts_stats["backpropagation"] += 1                           
                    iteration_history.append(batch_iteration.get(smin,MCTS_ITERATIONS))
                stats["docked"] += docked_count
            else:
                for smin in batch_smiles:
                    mcts_node =batch_mcts_nodes.get(smin)
                    if mcts_node is not None:
                        backpropagate_mcts(mcts_node, MCTS_REJECTION_REWARD)
                        mcts_stats["backpropagation"]+=1
        else:
            for smin in batch_smiles:
                mcts_node =batch_mcts_nodes.get(smin)
                if mcts_node is not None:
                    backpropagate_mcts(mcts_node, MCTS_REJECTION_REWARD)
                    mcts_stats["backpropagation"]+=1
            out_path=None
    result_df = pd.DataFrame(valid_compounds)
    output_csv = os.path.join(OUTPUT_DIR,"lstm_molecules.csv")
    else:
        pd.DataFrame(
            columns=["smiles","MW","LogP","HBD","HBA","TPSA","RingCount","RotBonds","qed","sa_score","lstm_confidence","inhibitor_probability","dock_score",
                "dock_normalized","novelty","gnn_component","dock_component","qed_component","sa_component","novelty_component","reward","iteration"]).to_csv(output_csv,index=False)

    all_generated_path = os.path.join(OUTPUT_DIR,"all_generated_smiles.csv")
    pd.DataFrame({"SMILES": generated_smiles_all}).to_csv(all_generated_path,index=False)
    generated_stats_file = os.path.join(OUTPUT_DIR,"generated_statistics.txt")


    n_generated = len(generated_smiles_all)
    for smi in generated_smiles_all:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(Chem.MolToSmiles(mol,canonical=True))
    n_valid = len(valid_smiles)
    unique_valid_smiles = list(dict.fromkeys(valid_smiles))
    n_unique_valid = len(unique_valid_smiles)
    unique_passed_smiles = list(dict.fromkeys(passed_smiles))
    n_final_accepted = len(passed_smiles)
    n_unique_final_accepted = len(unique_passed_smiles)
    n_passed_filters = n_final_accepted
    n_unique_passed = n_unique_final_accepted

    uniqueness_rate_valid = (n_unique_valid/n_valid if n_valid > 0 else 0.0)
    passed_rate_valid = (n_final_accepted/n_validif n_valid > 0 else 0.0)
    unique_passed_rate = (n_unique_passed/n_passed_filters if n_passed_filters > 0 else 0.0)

    def unique_at_k(smiles, k):
        subset = smiles[:min(k, len(smiles))]
        canonical_subset = []  
        for smi in subset:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical_subset.append(Chem.MolToSmiles(mol,canonical=True))
        return len(set(canonical_subset))

    unique_1000 = unique_at_k(valid_smiles,1000)
    novelty_values = [float(x["novelty"])
        for x in valid_compounds
        if x.get("novelty") is not None
        and not pd.isna(x.get("novelty"))]
    mean_novelty = (float(np.mean(novelty_values))if novelty_value else 0.0)

    gnn_values = [float(x["inhibitor_probability"])
        for x in valid_compounds
        if x.get("inhibitor_probability") is not None
        and not pd.isna(x.get( "inhibitor_probability"))]
    best_gnn_prob = (max(gnn_values) if gnn_values else 0.0)
    avg_gnn_prob = (float(np.mean(gnn_values))if gnn_values else 0.0)

    lstm_values = [float(x["lstm_confidence"])for x in valid_compounds
        if x.get("lstm_confidence") is not None and not pd.isna(x.get("lstm_confidence"))]
    best_lstm_confidence = (max(lstm_values)if lstm_values else 0.0)
    avg_lstm_confidence = (float(np.mean(lstm_values)) if lstm_values else 0.0)
    reward_values = [for x in valid_compounds if x.get("reward") is not None and not pd.isna(x.get("reward") )]
    best_rl_reward = (max(reward_values)if reward_values else 0.0)
    avg_rl_reward = (float(np.mean(reward_values))if reward_values else 0.0)

    for smi in unique_valid_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(morgan_gen.GetFingerprint(mol))
    if len(fps) >= 2:
        similarities = []
        max_pairs = 100000
        n_pairs = min(max_pairs, len(fps) * (len(fps) -1)//2)    
        for _ in range(n_pairs):
            i,j = random.sample(range(len(fps)),2)            
            sim = DataStructs.TanimotoSimilarity(fps[i],fps[j])
            similarities.append(sim)

        int_div = (1.0 -float(np.mean(similarities))
            if similarities
            else 0.0)
    else:
        int_div = 0.0

   return result_df


if __name__ == "__main__":

    LSTM_MODEL_PATH = "lstm_model_name"
    LSTM_VOCAB_PATH = "lstm_vocab_file"

    tokenizer = SMILESTokenizer.load(LSTM_VOCAB_PATH)
    stoi = tokenizer.token_to_id
    itos = tokenizer.id_to_token
    PAD_ID = stoi["[PAD]"]
    BOS_ID = stoi["[BOS]"]
    EOS_ID = stoi["[EOS]"]
    UNK_ID = stoi["[UNK]"]
    globals()["tokenizer"] = tokenizer
    globals()["stoi"] = stoi
    globals()["itos"] = itos
    globals()["PAD_ID"] = PAD_ID
    globals()["BOS_ID"] = BOS_ID
    globals()["EOS_ID"] = EOS_ID
    globals()["UNK_ID"] = UNK_ID

    checkpoint = torch.load(LSTM_MODEL_PATH,map_location=DEVICE)
    VOCAB_SIZE = checkpoint["vocab_size"]
    EMBED_DIM = checkpoint["embed_dim"]
    HIDDEN_DIM = checkpoint["hidden_dim"]
    NUM_LAYERS = checkpoint["num_layers"]
    DROPOUT = checkpoint["dropout"]
    MAX_LENGTH = checkpoint["max_length"]

    model = SMILESLSTM(vocab_size=VOCAB_SIZE,embed_dim=EMBED_DIM,hidden_dim=HIDDEN_DIM,num_layers=NUM_LAYERS,dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    for param in model.parameters():
        param.requires_grad_(False)
    globals()["model"] = model

    valid_test = 0
    for i in range(20):
        test_smiles, test_confidence = generate_smiles(model=model,temperature=TEMPERATURE,top_k=TOP_K,top_p=TOP_P,max_length=MAX_NEW_TOKENS)
        test_smiles_list.append(test_smiles)
        mol = Chem.MolFromSmiles(test_smiles)
        if mol is not None:
            valid_test += 1
            canonical_test = Chem.MolToSmiles(mol,canonical=True)

        else:
            print("INVALID")
    print("Test valid"    )

    train_df = pd.read_csv(CSV_FILE)
    train_smis = set(train_df[SMILES_COLUMN].dropna().astype(str).tolist())
    training_canonical = set()

    for smi in train_smis:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol,canonical=True)
                training_canonical.add(canonical)
        except Exception:
            continue
    morgan_gen = AllChem.GetMorganGenerator(radius=2,fpSize=2048)

    GNN_MODEL_PATH = "gnn_model_path"
    gnn_checkpoint = torch.load(GNN_MODEL_PATH,map_location=DEVICE)
    gnn_model = EdgeAwareGCNNet(in_channels=gnn_checkpoint["in_channels"],
        edge_dim=gnn_checkpoint["edge_dim"],hidden_dim=gnn_checkpoint["hidden_dim"]).to(DEVICE)
    gnn_model.load_state_dict(gnn_checkpoint["model_state_dict"])
    gnn_model.eval()

    for param in gnn_model.parameters():
        param.requires_grad_(False)

    feature_mean = gnn_checkpoint["feature_mean"]
    feature_std = gnn_checkpoint["feature_std"]
    results_df, summary = run_lstm_pipeline(model=model,train_smis=train_smis,gnn_model=gnn_model,
        feature_mean=feature_mean,feature_std=feature_std,morgan_gen=morgan_gen)