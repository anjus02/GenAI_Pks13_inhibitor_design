# GenAI_Pks13_inhibitor_design

# LSTM- and ChemGPT-Guided Molecular Generation Using MCTS

This repository implements two molecular generation strategies for identifying potential Pks13 inhibitors: **LSTM-guided Monte Carlo Tree Search (LSTM-MCTS)** and **ChemGPT-guided Monte Carlo Tree Search (ChemGPT-MCTS)**.

The LSTM model is trained on molecular sequences, while ChemGPT is fine-tuned on molecular representations to generate candidate compounds. Both models are integrated with Monte Carlo Tree Search (MCTS) to guide molecular exploration using a multi-objective reward function incorporating predicted Pks13 inhibitory activity, molecular docking scores, drug-likeness (QED), synthetic accessibility, and molecular novelty. Generated compounds are further evaluated using chemical validity checks and structural filtering criteria.

## Requirements

The `requirements.txt` file contains the Python libraries required to run the project. Install the dependencies using:

```bash
pip install -r requirements.txt
```

**Note:** External tools required for molecular docking, such as rDock, must be installed and configured separately.

## Repository Contents

| File | Description |
|---|---|
| `README.md` | Overview of the project, generation strategies, dependencies, and citation information |
| `LICENSE` | Specifies the licensing terms for using and distributing the repository |
| `requirements.txt` | Lists the Python libraries and versions used |
| `lstm_mcts.py` | Implements LSTM-guided Monte Carlo Tree Search (MCTS) |
| `lstm_model.py` | Defines the LSTM  model and its training |
| `chemgpt_direct.py` | Molecular generation using the fine-tuned ChemGPT model |
| `chemgpt_mcts.py` | ChemGPT-guided MCTS |
| `evaluation_scip.py` | Evaluation procedures for candidates and associated properties |
| `sascorer.py` | Provides functions for calculating synthetic accessibility scores using the Ertl–Schuffenhauer method. |
| `data.csv` | Pks13 Inhibitor data |
| `ligands.smi` | Lligand structures in SMILES format |
| `lstm_vocab.json` | LSTM-model vocab|
| `pks13_vocab.json` | ChemGPT-model vocab |
| `sa_qed_bin_edges.npz` | Bin-edge arrays used for SA and QED conditioning|
| `gnn_model.pt` | GNN model |
| `scaler.pt` | Feature-scaling object |
| `scheduler.pt` | Training scheduler checkpoint |
| `8Q17.pdb` | Protein structure in PDB format  |
| `IL8_ligand.pdb` | Ligand structure |


##  Citation
Please cite:
Sharma, A. et al. (2026). Expanding the Chemical Landscape of Pks13 Inhibitors Through Language-Based Deep Generative Modeling. [Manuscript under review]

##  Contact
For queries, suggestions, or collaborations:

### Dr. Anju Sharma
anjusharma.online@gmail.com
