#!/usr/bin/env python3
"""
MCTS Search with Fine‑tuned ChemGPT (chem_local)
Reward = Docking score + QED + SA score + Novelty
Filters = PAINS + SA + Drug-likeness
"""

import time, random, numpy as np, torch, json, os
from math import sqrt, log
from transformers import AutoModelForCausalLM
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors, Draw
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
import re
import glob
from rdkit import Chem, DataStructs
import selfies as sf
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import FilterCatalog
from training_advanced import EdgeAwareGCNNet
from Smiles_to_graph import smiles_to_graph
import json
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GENERATED = 1000
MCTS_REJECTION_REWARD = -0.10

print("Using device:", DEVICE)

def graph_to_feature_vector(graph):
    x = graph.x.detach().cpu().numpy()
    if x.shape[0] == 0:
        raise ValueError("Graph contains no atoms.")
    mean_features = np.mean(x, axis=0)
    std_features = np.std(x, axis=0)
    return np.concatenate([mean_features, std_features])
              
def fit_ad_model(train_smis):
    input_features = []
    for i, smiles in enumerate(train_smis):
        features = smiles_to_ad_features(smiles)
        if features is not None:
            if i ==0:
                graph = smiles_to_graph(str(smiles))
            input_features.append(features)
    X_input = np.asarray(input_features, dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_input)
    variance = np.var(X_scaled, axis=0)
    keep_features = variance > 0
    X_filtered = X_scaled[:, keep_features]

    lw = LedoitWolf()
    lw.fit(X_filtered)

    mean_vector = lw.location_
    precision_matrix = lw.precision_
    diff = X_filtered - mean_vector
    distance_squared = np.einsum("ij,jk,ik->i",diff,precision_matrix,diff)
    distance_squared = np.maximum(distance_squared,0)
    input_distances = np.sqrt(distance_squared)
    ad_threshold = np.percentile(input_distances,95)

    return {"scaler": scaler,"keep_features": keep_features,"mean_vector": mean_vector,"precision_matrix": precision_matrix,"threshold": ad_threshold}
    
def predict_ad(smiles,ad_model):
    features = smiles_to_ad_features(smiles)
    if features is None:
        return None, None
    features = np.asarray(features,dtype=float).reshape(1, -1)
    scaler = ad_model["scaler"]
    keep_features = ad_model["keep_features"]
    mean_vector = ad_model["mean_vector"]
    precision_matrix = ad_model["precision_matrix"]
    threshold = ad_model["threshold"]
    scaled = scaler.transform(features)
    scaled = scaled[:, keep_features]
    diff = scaled - mean_vector
    distance_squared = np.einsum("ij,jk,ik->i",diff,precision_matrix,diff)
    distance_squared = np.maximum(distance_squared,0)
    distance = float(np.sqrt(distance_squared)[0])
    inside = distance <= threshold
    return distance, inside
    
def predict_smile(model, tokenizer, state_text):
    prompt = f"<SA_BIN_4><QED_BIN_7>{tokenizer.bos_token}"
    input_ids = torch.tensor([tokenizer.encode(prompt)],device=model.device)
    outputs = model.generate(input_ids=input_ids, min_new_tokens=20,max_new_tokens=150,do_sample=True,temperature=1.2,   #0.8,
        top_p=0.97,top_k=80,repetition_penalty=1.15,eos_token_id=None)

    raw_text = tokenizer.decode(outputs[0].tolist())

    selfies = raw_text
    selfies = re.sub(r"<SA_BIN_\d+>", "", selfies)
    selfies = re.sub(r"<QED_BIN_\d+>", "", selfies)
    selfies = selfies.replace("[UNK]", "")
    selfies = selfies.replace("[BOS]", "")
    selfies = selfies.replace("[EOS]", "")
    selfies = "".join(re.findall(r"\[[^\]]+\]", selfies))   

    try:
        smi = sf.decoder(selfies)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print("Invalid SMILES")
            return None
        smi = Chem.MolToSmiles(mol)
        return smi
    except Exception as e:
        print("SELFIES decode failed:", e)
        return None

def validate_selfies_and_smiles(selfies_str):
    try:
        smi = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return mol, smi
        return (mol, smi) if mol else (None, None)
    except:
        return None, None

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
        self.pad_id = self.pad_token_id
        self.eos_id = self.eos_token_id
    
    def __len__(self):
        return len(self.vocab)
    
    def __call__(self, text, truncation=True, max_length=384, padding=False, add_special_tokens=False, return_tensors=None):
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(20, len(text)-i), 0, -1):
                candidate = text[i:i+length]
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
        attention_mask = [1] * len(input_ids)      
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    def encode(self, text, *args, **kwargs):
        return self(text, *args, **kwargs)["input_ids"]
    
    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [self.ids_to_tokens.get(tid, self.unk_token) for tid in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in [self.pad_token, self.bos_token, self.eos_token]]
        return ''.join(tokens)

def load_model(adapter_dir="finetuned_chemgpt"):
    vocab = json.load(open(f"{adapter_dir}/pks13_vocab.json"))
    tokenizer = AcheTokenizer(vocab)
   
    model = AutoModelForCausalLM.from_pretrained("{adapter_dir}/merged_model",dtype=torch.float16,device_map="auto")
    model.eval()   
    generated = []
    
    for i in range(100):
        smi = predict_smile(model, tokenizer, "<SA_BIN_4><QED_BIN_7>")
        print(f"{i+1}: {smi}")       
        if smi is not None:
            generated.append(smi)
    return model, tokenizer

def smiles_list_to_sdf(smiles_list, out_file="batch_ligands_246_temp1.2_bs50.sdf"):
    writer = Chem.SDWriter(out_file)
    written = 0
    failed = 0
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"[SDF] Invalid SMILES: {smi}")
                failed += 1
                continue
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.useRandomCoords = True
            try:
                result = AllChem.EmbedMolecule(mol, params)
            except RuntimeError as e:
                print(f"[SDF] Embedding failed: {smi}")
                print(f"{str(e).splitlines()[0]}")
                failed += 1
                continue
            if result != 0:
                print(f"[SDF] Could not embed: {smi}")
                failed += 1
                continue
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol,maxIters=200)
                else:
                    AllChem.UFFOptimizeMolecule(mol,maxIters=200)

            except Exception:
                print(f"[SDF] Optimization failed: {smi}")
           
            mol.SetProp("SMILES", smi)
            mol.SetProp("MCTS_INDEX", str(i))          
            writer.write(mol)
            written += 1
        except Exception as e:
            print(f"       {e}")
            failed += 1
            continue
    writer.close()
    print("failed")
    if written == 0:
        return None
    return out_file

def run_batch_docking(sdf_path, receptor="dock_receptor.prm", protocol="dock.prm", out_path="batch_out"):
    cmd = ["path_to_rbdock","-r", receptor,"-p", protocol,"-i", sdf_path,"-o", out_path,"-n", "20"]
    result = subprocess.run(cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError("rDock failed") 
    out_file = out_path + ".sd" 
    if not os.path.exists(out_file):
        raise FileNotFoundError(f"{out_file} was not created by rDock")   
    return out_file

def parse_docking_scores(sd_file, input_smiles):
    input_set = set(str(s).strip() for s in input_smiles)
    try:
        with open(sd_file, "r") as f:
            content = f.read()
        records = content.split("$$$$")
        total_records = 0
        score_records = 0
        matched_records = 0
        unmatched_records = 0

        for record in records:
            if not record.strip():
                continue
            total_records += 1
            lines = record.splitlines()
            smiles = None
            score = None
            for i, line in enumerate(lines):
                tag = line.strip()
                if tag in [">  <SMILES>", "> <SMILES>"]:
                    for j in range(i + 1, len(lines)):
                        value = lines[j].strip()
                        if value:
                            smiles = value
                            break
                elif tag in [">  <SCORE>", "> <SCORE>"]:
                    for j in range(i + 1, len(lines)):
                        value = lines[j].strip()
                        if value:
                            try:
                                score = float(value)
                            except ValueError:
                                score = None
                            break
            if score is not None:
                score_records += 1
            if smiles is not None and score is not None:
                if smiles in input_set:
                    if smiles not in docking_scores:
                        docking_scores[smiles] = []
                    docking_scores[smiles].append(score)
                    matched_records += 1
                else:
                    unmatched_records += 1
            else:
                if score is not None:
                    unmatched_records +=
        return {}
   
def docking_score(smiles, receptor="dock_receptor.prm", protocol="dock.prm", out_path="dock_out.sd"):
    sdf_path = smiles_to_sdf(smiles)
    if sdf_path is None:
        return None
    cmd = ["path_to_rbdock","-r", receptor, "-p", protocol, "-i", sdf_path, "-o", out_path, "-n", "20"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return None
    score = None
    with open(out_path) as f:
        for line in f:
            if line.strip().startswith("> <SCORE>"):
                next_line = next(f).strip()
                try:
                    score = float(next_line)
                except:
                    pass
                break
    return score

def pains_filter(mol):
    if mol is None:
        return False
    return not PAINS_CATALOG.HasMatch(mol)

def synthetic_accessibility(mol):
    try:
        mw = Descriptors.MolWt(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        return 10.0 - (mw/100.0 + rings)
    except:
        return 0.0

def strict_drug_like_filter(mol):
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        halogen_count = sum(1
            for atom in mol.GetAtoms()
            if atom.GetSymbol() in {"F", "Cl", "Br", "I"})

        double_bonds = sum(1
            for bond in mol.GetBonds()
            if bond.GetBondType() == Chem.BondType.DOUBLE)

        if not (150.0 <= mw <= 600.0):
            return False
        if logp > 5.0:
            return False
        if hbd > 5:
            return False
        if hba > 12:
            return False
        if halogen_count > 2:
            return False
        if double_bonds > 8:
            return False
        if rot_bonds > 12:
            return False
        if heavy_atoms < 12 or heavy_atoms > 70:
            return False
        return True
    except Exception:
        return False

def compute_properties(mol, smi):
    return {"smiles": smi,"mw": Descriptors.MolWt(mol),"logp": Descriptors.MolLogP(mol),"hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),"rtb": Lipinski.NumRotatableBonds(mol), "qed": QED.qed(mol),"num_rings": rdMolDescriptors.CalcNumRings(mol)}

class Chemical:
    def __init__(self):
        self.position = ['C']
    def Clone(self):
        st = Chemical()
        st.position = self.position[:]
        return st
    def SelectPosition(self, m):
        self.position.append(m)

from math import sqrt, log
class Node:
    def __init__(self, position=None, parent=None, state=None):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0

    def Selectnode(self):
        best_score = -float("inf")
        best_child = None    
        for child in self.childNodes:         
            if child.visits ==0:
                return child           
            exploitation = child.wins/child.visits
            exploration = sqrt(2.0*log(max(1,self.visits))/child.visits)        
            score = exploitation + exploration   
            if score > best_score:
                best_score = score
                best_child = child              
        return best_child

    def Addnode(self, m, s):
        n = Node(position=m, parent=self, state=s)
        self.childNodes.append(n)
        return n
    def Update(self, result):
        self.visits += 1
        self.wins += result

def backpropagate_reward(node, reward):
    current = node
    while current is not None:
        current.Update(float(reward))
        current = current.parentNode

def novelty(smiles):
    if smiles in seen_smiles:
        return -1.0
    else:
        seen_smiles.add(smiles)
        return 1.0

def relaxed_ring_check(mol):
    try:
        rings = mol.GetRingInfo().AtomRings()
        if not rings:
            return True
        if any(len(ring) < 3 or len(ring) > 7 for ring in rings):
            return False
        if len(rings) > 5:
            return False
        fused_pairs = 0
        for i, ring_i in enumerate(rings):
            atoms_i = set(ring_i)
            for ring_j in rings[i + 1:]:
                if len(atoms_i.intersection(ring_j)) >= 2:
                    fused_pairs += 1
                    if fused_pairs > 1:
                        return False
        return True
    except Exception:
        return False


def predict_gnn_inhibitor(smiles, gnn_model, feature_mean, feature_std):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    graph = smiles_to_graph(smiles, 0.0)
    if graph is None:
        return None
    device = next(gnn_model.parameters()).device
    graph = graph.to(device)
    feature_mean = torch.as_tensor(feature_mean,dtype=graph.x.dtype,device=device)
    feature_std = torch.as_tensor(feature_std,dtype=graph.x.dtype,device=device)
    feature_std = torch.where(feature_std == 0,torch.ones_like(feature_std),feature_std)
    graph.x = (graph.x - feature_mean) / feature_std

    gnn_model.eval()
    with torch.no_grad():
        out, _ = gnn_model(graph.x,graph.edge_index,graph.edge_attr,batch)
        probability = torch.sigmoid(out).view(-1)[0].item()
    return float(probability)

W_GNN = 0.40
W_DOCK = 0.30      
W_QED = 0.10     
W_SA = 0.10        
W_NOVELTY = 0.10   
GNN_THRESHOLD = 0.60
QED_MIN = 0.40    
SA_MAX = 7.0     
SA_MIN = 1.0
seen_smiles = set()
all_generated_smiles = []
passed_smiles = []

def MCTS(root, model, tokenizer, train_smis, train_fps, gnn_model, feature_mean, feature_std, ad_model, output_dir="mcts_results_246_temp1.2_bs50"):
    os.makedirs(output_dir, exist_ok=True)
    rootnode = Node(state=root)
    valid_compound = []
    batch_smiles = []    
    batch_size = 32
    best_reward_history = []
    iteration_history = []
    reward_history = []
    reward_event_history = []
    best_reward_so_far = -float("inf")
    iteration = 0
    reward_event = 0
    
    stats = {"generated": 0,"valid": 0,"ring_pass": 0, "druglike": 0,"non_pains": 0,"gnn_pass": 0,"embedded": 0,"docked": 0,"qed_pass":0, "sa_pass":0, "accepted": 0}
    gnn_probabilities = {}
    ad_distances = {}
 
    while stats["generated"] < N_GENERATED:
        iteration += 1
        node = rootnode
        state = root.Clone()
        while node.childNodes:
            node = node.Selectnode()
            state.SelectPosition(node.position)
        state_text = "<SA_BIN_4> <QED_BIN_7> "      
        smi = predict_smile(model, tokenizer, state_text)
        
        if smi is None:
            backpropagate_reward(node, MCTS_REJECTION_REWARD)
            continue      
        all_generated_smiles.append(smi)

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            backpropagate_reward(node, MCTS_REJECTION_REWARD)
            continue
     
        if smi in train_smis:
            backpropagate_reward(node, MCTS_REJECTION_REWARD)
            continue
            
        if smi in seen_smiles:
            backpropagate_reward(node, MCTS_REJECTION_REWARD)
            continue
        
        seen_smiles.add(smi)
        
        if not relaxed_ring_check(mol):
            backpropagate_reward(node, MCTS_REJECTION_REWARD)
            continue
        
        if not strict_drug_like_filter(mol):
            print("Rejected: strict drug-like")
            backpropagate_reward(node, MCTS_REJECTION_REWARD)
            continue

        if not pains_filter(mol):
            backpropagate_reward(node, MCTS_REJECTION_REWARD)
            continue
        
        inhibitor_prob = predict_gnn_inhibitor(smi, gnn_model, feature_mean, feature_std)
        if inhibitor_prob is None:
            backpropagate_reward(node, MCTS_REJECTION_REWARD)
            continue

        if inhibitor_prob < GNN_THRESHOLD:
            print("Rejected:)
            backpropagate_reward(node, MCTS_REJECTION_REWARD)
            continue

        gnn_probabilities[smi] = inhibitor_prob
        ad_distance, ad_inside = predict_ad(smi, ad_model)
        ad_distances[smi] = ad_distance      
        if ad_inside:
            stats["ad_pass"] += 1
        else:
            print(fOutside AD")

        passed_smiles.append(smi)
        batch_smiles.append(smi)
        
        if len(batch_smiles) < batch_size:
            continue          
        sdf_path = smiles_list_to_sdf(batch_smiles)              
        if sdf_path is None:
            batch_smiles = []
            continue      
        out_path = run_batch_docking(sdf_path)
        
        if not os.path.exists(out_path):
            batch_smiles = []
            continue
                    
        scores = parse_docking_scores(out_path, batch_smiles)
        print ("Scores =", scores)
        docked_count =0

        for idx, smin in enumerate(batch_smiles):
            mol = Chem.MolFromSmiles(smin)
            if mol is None:
                continue
           
            molecules_scores = scores.get(smin)                           
            dock = min(molecules_scores)      
 
            inhibitor_prob = gnn_probabilities.get(smin)
            if inhibitor_prob is None:
                print("NO GNN probability")
                continue
            
            ad_distance = ad_distances.get(smin)
            if ad_distance is None:
                print ("AD missing")
                continue
            
            sa = synthetic_accessibility(mol)
            qed = QED.qed(mol)           

            if qed < QED_MIN:
                print("Rejected by QED:")
                backpropagate_reward(node, MCTS_REJECTION_REWARD)
                continue
            
            if sa > SA_MAX:
                print("Rejected by SA")
                backpropagate_reward(node, MCTS_REJECTION_REWARD)
                continue         
                    
            props = compute_properties(mol,smin)
            props["dock_score"] = dock
            props["sa_score"] = sa
            props["qed"] = qed
            props["inhibitor_probability"] = inhibitor_prob
            props["AD_distance"] = ad_distance
            props["AD_threshold"] = ad_model["threshold"]
            props["AD_status"] = ("Inside_AD"
                if ad_distance <= ad_model["threshold"]
                else "Outside_AD")
      
            fp = morgan_gen.GetFingerprint(mol)
            if len(train_fps) >0:
                max_sim = max(DataStructs.TanimotoSimilarity(fp, x)
                for x in train_fps
            )
            else:
                max_sim = 0.0
                
            novelty = 1.0 - max_sim
            props["novelty"] = novelty      
            sa_norm = 1.0 - (sa - SA_MIN) /(SA_MAX - SA_MIN)
            sa_norm = np.clip(sa_norm, 0.0, 1.0)
            gnn_component = W_GNN * inhibitor_prob
            dock_component = W_DOCK * dock_norm
            qed_component = W_QED * qed
            sa_component = W_SA * sa_norm
            novelty_component = W_NOVELTY * novelty
            reward = (gnn_component + dock_component + qed_component + sa_component +  novelty_component)
            
            reward_history.append(reward)           
            if reward > best_reward_so_far:
                best_reward_so_far = reward
            
            reward_event += 1
            
            iteration_history.append(iteration)
            reward_event_history.append(reward_event)
            best_reward_history.append(best_reward_so_far)
  
            props["gnn_component"] = gnn_component
            props["dock_component"] = dock_component
            props["dock_normalized"] = dock_norm
            props["qed_component"] = qed_component
            props["sa_normalized"] = sa_norm
            props["sa_component"] = sa_component
            props["novelty_component"] = novelty_component
            props["reward"] = reward
            
            valid_compound.append(props)
            stats["accepted"] += 1           

            child = node.Addnode(smin, state)
            while child is not None:
                child.Update(reward)
                child = child.parentNode
        
        stats["docked"] += docked_count        
        print("Docking scores obtained")
        batch_smiles =[]       

    if len(batch_smiles) > 0:  
        gnn_passed_smiles = []
        for smin in batch_smiles:
            inhibitor_prob = gnn_probabilities.get(smin)  
            if inhibitor_prob is None:
                print("GNN prediction faile")
                continue
            print"GNN Prediction done")
            
            if inhibitor_prob < GNN_THRESHOLD:
                print ("Rejected by GNN")
                continue                
            gnn_passed_smiles.append(smin)
        
        if len(gnn_passed_smiles) == 0:
            print ("no molecules passed GNN")
        else:
            sdf_path = smiles_list_to_sdf(gnn_passed_smiles)           
            if sdf_path is not None:
                out_path = run_batch_docking(sdf_path)
                if out_path is not None and os.path.exists(out_path):
                    scores = parse_docking_scores(out_path, gnn_passed_smiles) 
                    docked_count = 0
                    for idx, smin in enumerate(gnn_passed_smiles):
                        mol = Chem.MolFromSmiles(smin)
                        if mol is None:
                            continue      
                       
                        molecule_scores = scores.get(smin)
                        if molecule_scores is None:
                            print("No docking record")
                            continue
                        
                        if len(molecule_scores) == 0:
                            print("No docking")
                            continue
                        
                        dock = min(molecule_scores)                   
                        if dock is None:
                            print ("No docking")
                            continue
                        
                        inhibitor_prob = gnn_probabilities.get(smin)
                        if inhibitor_prob is None:
                            print("GNN prediction failed")
                            continue
                        
                        ad_distance = ad_distances.get(smin)
                        if ad_distance is None:
                            print("AD distance miss")
                            continue

                        qed = QED.qed(mol)
                        sa = synthetic_accessibility(mol)

                        if qed < QED_MIN:
                            print ("Rejected by QED: "
                                f"{qed:.3f} < {QED_MIN}")
                            continue
            
                        if sa > SA_MAX:
                            print("Rejected by SA")
                            continue
                    
                        fp = morgan_gen.GetFingerprint(mol)
                        if len(train_fps) >0:
                            max_sim = max( DataStructs.TanimotoSimilarity(fp, x) for x in train_fps)
                        else:
                            max_sim = 0.0
 
                        novelty = 1.0 - max_sim   

                        props = compute_properties(mol, smin)
                        props["sa_score"] =sa
                        props["dock_score"] = dock
                        props["qed"] = qed
                        props["inhibitor_probability"] = inhibitor_prob
                        props["AD_distance"] = ad_distance
                        props["AD_threshold"] = ad_model["threshold"]
                        props["AD_status"] = ("Inside_AD" if ad_inside else "Outside_AD")                     
                        props["novelty"] = novelty
                        sa_norm = 1.0 - (sa - SA_MIN) /(SA_MAX - SA_MIN)
                        sa_norm = np.clip(sa_norm, 0.0, 1.0)

                        gnn_component = W_GNN * inhibitor_prob
                        dock_component = W_DOCK * dock_norm
                        qed_component = W_QED * qed
                        sa_component = W_SA * sa_norm
                        novelty_component = W_NOVELTY * novelty
                        reward = (gnn_component + dock_component + qed_component + sa_component +  novelty_component)
                        
                        reward_history.append(reward)
            
                        if reward > best_reward_so_far:
                            best_reward_so_far = reward
                        
                        reward_event += 1
                        iteration_history.append(iteration)
                        reward_event_history.append(reward_event)
                        best_reward_history.append(best_reward_so_far)

                        props["gnn_component"] = gnn_component
                        props["dock_component"] = dock_component
                        props["dock_normalized"] = dock_norm
                        props["qed_component"] = qed_component
                        props["sa_normalized"] = sa_norm
                        props["sa_component"] = sa_component
                        props["novelty_component"] = novelty_component            
                        props["reward"] = reward
                        valid_compound.append(props)
                        stats["accepted"] += 1    

                        child =node.Addnode(smin,state)
                        while child is not None:
                            child.Update(reward)
                            child=child.parentNode                              
                    stats["docked"]+= docked_count
                else:
                    print ("Docking failed")       
            else:
                print("No valid molceuls remained for final docking abtch")
            
    batch_smiles = []          
        
    df = pd.DataFrame(valid_compound)
    output_csv = os.path.join(output_dir, "mcts_molecules.csv")  
    return valid_compound

if __name__ == "__main__":
    model, tokenizer = load_model("finetuned_chemgpt")
    state = Chemical()
    train_df = pd.read_csv("pks13_smiles.csv")
    train_smis = set(train_df["SMILES"].dropna().tolist())
    ad_model = fit_ad_model(train_smis)
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    train_fps = [
        morgan_gen.GetFingerprint(Chem.MolFromSmiles(s))
        for s in train_smis
        if Chem.MolFromSmiles(s) is not None
    ]

    GNN_MODEL_PATH = "gnn_model.pt"
    checkpoint = torch.load(GNN_MODEL_PATH,map_location=DEVICE)
    gnn_model = EdgeAwareGCNNet(in_channels=checkpoint["in_channels"],edge_dim=checkpoint["edge_dim"],hidden_dim=checkpoint["hidden_dim"]).to(DEVICE)
    gnn_model.load_state_dict(checkpoint["model_state_dict"])
    gnn_model.eval()
    feature_mean = checkpoint["feature_mean"]
    feature_std = checkpoint["feature_std"]
    
    results = MCTS(state,model,tokenizer,train_smis,train_fps,gnn_model,feature_mean,feature_std, ad_model,output_dir="mcts_results_246_temp1.2_bs50")
