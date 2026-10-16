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

##  Citation
Please cite:
Sharma, A. et al. (2026). Expanding the Chemical Landscape of Pks13 Inhibitors Through Language-Based Deep Generative Modeling. [Manuscript under review]

##  Contact
For queries, suggestions, or collaborations:

### Dr. Anju Sharma
anjusharma.online@gmail.com
