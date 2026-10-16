#!/usr/bin/env python3
import os
import re
import json
import glob
import subprocess
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import (Descriptors,Lipinski,QED,rdMolDescriptors,FilterCatalog,AllChem)
from training_advanced import EdgeAwareGCNNet
from Smiles_to_graph import smiles_to_graph

DIRECT_CSV = "output_file"
TRAIN_CSV = "input_filev"
TRAIN_SMILES_COLUMN = "SMILES"
OUTPUT_DIR = "output_directory"
GNN_MODEL_PATH = "gnn_model_path"
GNN_THRESHOLD = 0.60
SEED = 42
DOCK_BATCH_SIZE = 50
RDOCK_N_POSES = 20
RDOCK_BIN = ("path_rbdock")
RECEPTOR = "dock_receptor.prm"
PROTOCOL = "dock.prm"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpuu")
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def canonicalize_smiles(smiles):
    try:
        if smiles is None or pd.isna(smiles):
            return None
        smiles = str(smiles).strip()
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol,canonical=True,)
    except Exception:
        return None

def get_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048,)

def calculate_novelty(mol, train_fps):
    fp = get_fingerprint(mol)
    if not train_fps:
        return 1.0
    max_similarity = max(DataStructs.TanimotoSimilarity(fp,train_fp,) for train_fp in train_fps)
    return float(1.0 - max_similarity)

def ring_filter(mol):
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
                shared_atoms = len(atoms_i.intersection(set(ring_j)))
                if shared_atoms >= 2:
                    fused_pairs += 1
                    if fused_pairs > 1:
                        return False
        return True
    except Exception:
        return False

def physicochemical_filter(mol):
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        halogen_count = sum(atom.GetSymbol() in {"F", "Cl", "Br", "I"}
            for atom in mol.GetAtoms())
        double_bonds = sum(bond.GetBondType() == Chem.BondType.DOUBLE
            for bond in mol.GetBonds())
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
        if not (12 <= heavy_atoms <= 70):
            return False
        return True
    except Exception:
        return False

def pains_filter(mol):
    try:
        return len(pains_catalog.GetMatches(mol)) == 0
    except Exception:
        return False


def load_gnn_model():
    checkpoint = torch.load(GNN_MODEL_PATH,map_location=DEVICE,weights_only=False,)
    model = EdgeAwareGCNNet(in_channels=checkpoint["in_channels"],edge_dim=checkpoint["edge_dim"],hidden_dim=checkpoint["hidden_dim"],).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return (model,checkpoint["feature_mean"],checkpoint["feature_std"],)

def predict_gnn_inhibitor(smiles,gnn_model,feature_mean,feature_std,):
    try:
        mol = Chem.MolFromSmiles(smiles)
        graph = smiles_to_graph(smiles,0.0,)
        if graph is None:
            return None
        device = next(gnn_model.parameters()).device
        graph = graph.to(device)
        mean = torch.as_tensor(feature_mean,dtype=graph.x.dtype,device=device)
        std = torch.as_tensor(feature_std,dtype=graph.x.dtype,device=device)
        std = torch.where(std == 0,torch.ones_like(std),std)
        graph.x = (graph.x - mean) / std
        with torch.no_grad():
            out, _ = gnn_model(graph.x,graph.edge_index,graph.edge_attr,batch,)
            probability = torch.sigmoid(out).view(-1)[0].item()
        return float(probability)
    except Exception as exc:
        return None

def smiles_list_to_sdf(smiles_list,out_file,):
    writer = Chem.SDWriter(out_file)
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = SEED
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol,params,)
            if result != 0:
                continue
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol,maxIters=200,)
                else:
                    AllChem.UFFOptimizeMolecule(mol,maxIters=200,)
            except Exception:
                pass
            mol.SetProp("SMILES",smi,)
            mol.SetProp("MCTS_INDEX",str(i),)
            writer.write(mol)
            written += 1
        except Exception as exc:
    writer.close()
    return (out_file if written > 0 else None)

def run_batch_docking(sdf_path,out_prefix,):
    cmd = [RDOCK_BIN,"-r",RECEPTOR,"-p",PROTOCOL,"-i",sdf_path,"-o",out_prefix,"-n",str(RDOCK_N_POSES),]
    result = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,)
    if result.returncode != 0:
        return None
    candidates = [out_prefix + ".sd",out_prefix + ".sdf",out_prefix,]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def parse_docking_scores(sd_file,batch_smiles,):
    input_canonical = {}
    for smi in batch_smiles:
        csmi = canonicalize_smiles(smi)
        if csmi is not None:
            input_canonical[csmi] = smi
    docking_scores = defaultdict(list)
    with open(sd_file,"r",errors="ignore",) as f: text = f.read()
    records = text.split("$$")
    for record in records:
        if "<SCORE>" not in record:
            continue
        score_match = re.search(r">\<SCORE>\n"r"\([-+]?\\.?\d+(?:[Ee][-+]?\d+)?)",record)
        if score_match is None:
            continue
        try:
            score = float(score_match.group(1))
        except Exception:
            continue
        smiles_match = re.search( r">\<SMILES>\n"r"(.+)",record,)
        if smiles_match is None:
            continue
        dock_smiles = (smiles_match.group(1).strip())
        canonical_dock = canonicalize_smiles(dock_smiles)
        if canonical_dock in input_canonical:
            original = input_canonical[canonical_dock]
            docking_scores[original].append(score)
    return dict(docking_scores)

direct_df = pd.read_csv(DIRECT_CSV)
n_attempts = len(direct_df)
if n_attempts != 1000:
    print("WARNING: direct-generation failur")
if "canonical_smiles" in direct_df.columns:
    source_column = "canonical_smiles"
elif "raw_smiles" in direct_df.columns:
    source_column = "raw_smiles"
elif "smiles" in direct_df.columns:
    source_column = "smiles"
else:
    raise ValueError("No SMILES column.")

gnn_model, feature_mean, feature_std = (load_gnn_model())
gnn_probs = []
for i, smi in enumerate(structural_pass_df["smiles"],start=1,):
    prob = predict_gnn_inhibitor(smi,gnn_model,feature_mean,feature_std,)
    gnn_probs.append(prob)
structural_pass_df[ "inhibitor_probability"] = gnn_probs
gnn_evaluated_df = (structural_pass_df[structural_pass_df["inhibitor_probability"].notna()].copy())
gnn_pass_df = (gnn_evaluated_df[gnn_evaluated_df["inhibitor_probability"] >= GNN_THRESHOLD].copy())
n_gnn_evaluated = len(gnn_evaluated_df)
n_gnn_pass = len(gnn_pass_df)
gnn_pass_rate_structural = (n_gnn_pass /n_gnn_evaluated if n_gnn_evaluated > 0else 0.0)

all_docking_records = []
gnn_pass_smiles = (gnn_pass_df["smiles"].tolist())
for batch_index, start in enumerate(range(0,len(gnn_pass_smiles),DOCK_BATCH_SIZE,),start=1):
    batch = gnn_pass_smiles[start:start + DOCK_BATCH_SIZE]
    if not batch:
        continue
    sdf_path = os.path.join(OUTPUT_DIR,"name_file.sdf")
    out_prefix = os.path.join( OUTPUT_DIR,"batch_file")
    sdf_path = smiles_list_to_sdf(batch,sdf_path)
    if sdf_path is None:
        continue
    docking_output = run_batch_docking(sdf_path,out_prefix)
    if docking_output is None:
        continue
    scores = parse_docking_scores(docking_output,batch)
    for smi in batch:
        molecule_scores = scores.get(smi,[])
        if not molecule_scores:
            continue
        all_docking_records.append({"smiles": smi,"n_poses":len(molecule_scores),"dock_score":min(molecule_scores)}
docking_df = pd.DataFrame(all_docking_records)
if not docking_df.empty:
    docked_df = gnn_pass_df.merge(docking_df,on="smiles",how="inner")
else:
    docked_df = gnn_pass_df.iloc[0:0].copy()
    docked_df["n_poses"] = pd.Series(dtype=int)
    docked_df["dock_score"] = pd.Series(dtype=float)

fps = []
for smi in unique_valid_df["evaluation_smiles"]:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fps.append(get_fingerprint(mol))
if len(fps) >= 2:
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1,len(fps)):
            similarities.append(
                DataStructs.TanimotoSimilarity(fps[i],fps[j]))
    internal_diversity = float(1.0 - np.mean(similarities))
else:
    internal_diversity = 0.0
def safe_mean(series):
    series = pd.to_numeric(series,errors="coerce",).dropna()
    return (float(series.mean())if len(series)else None)

def safe_median(series):
    series = pd.to_numeric(series,errors="coerce").dropna()
    return (float(series.median())if len(series)else None)

def safe_std(series):
    series = pd.to_numeric(series,errors="coerce",).dropna()
    return (float(series.std(ddof=1))if len(series) > 1 else None)

def safe_iqr(series):series = pd.to_numeric(series,errors="coerce",).dropna()
    if len(series) == 0:
        return None
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    return q3 - q1

for key, value in summary.items():
    print(f"{key:45s}: {value}")
