# VACS: Value-Aligned Compositional Shielding for Multi-Agent Reasoning

This repository contains the implementation and experimental code for the paper **"VACS: Value-Aligned Compositional Shielding for Multi-Agent Reasoning"**.

VACS is a four-layer framework designed to align multi-agent reasoning systems with heterogeneous value priorities. It integrates:
1.  **Value Profiling**: Inferring agent values from behavior.
2.  **Compositional Shielding**: Enforcing safety constraints via assume-guarantee reasoning.
3.  **Hamiltonian Consensus**: Resolving conflicts using nucleolus-weighted optimization.
4.  **Faithful Explanation**: Generating explanations grounded in the formal proof structure.

## Repository Structure

```
VACS_MultiAgentReasoning/
├── data/                   # Benchmark datasets
│   ├── math_subset.csv     # MathInstruct (Reasoning) subset (N=200)
│   └── nejm_qa.csv         # NEJM-AI QA (Clinical) benchmark
├── experiments/            # Experiment scripts
│   ├── run_vacs.py         # Main entry point for VACS experiments
│   ├── ablation_study.py   # Script to reproduce ablation results (Table 3)
│   └── run_baselines.py    # Baseline methods (Majority Vote, etc.)
├── vacs/                   # Core framework implementation
│   ├── layer1_...py        # Value profiling & IRL
│   ├── layer2_...py        # Shielding & Safety constraints
│   ├── layer3_...py        # Consensus & Nucleolus calculation
│   ├── layer4_...py        # Explanation generation
│   └── utils.py            # Helpers (including robust answer normalization)
└── paper1_VACS.tex         # LaTeX source for the paper
```

## Requirements

The framework is implemented in Python. Key dependencies include:

*   `numpy`
*   `pandas`
*   `scipy` (for optimization/nucleolus calculation)
*   `tqdm` (for progress bars)

Install dependencies via pip:
```bash
pip install numpy pandas scipy tqdm
```

## Running Experiments

### 1. Main VACS Performance
To evaluate VACS on the **MathInstruct** and **NEJM-AI QA** benchmarks and see detailed case outputs:

```bash
python experiments/run_vacs.py
```

### 2. Ablation Study
To reproduce the ablation results (Table 3 in the paper), which analyze the contribution of each layer (Profiling, Shielding, Consensus, Explanation):

```bash
python experiments/ablation_study.py
```
*This script runs VACS in various configurations (e.g., `no_layer1`, `no_layer2`) and reports Accuracy and Logical Inconsistency Rate (LIR).*

## Datasets

The repository includes processed versions of the benchmarks used in the paper:
*   **MathInstruct-Subset**: A challenging mathematical reasoning dataset (N=200).
*   **NEJM-AI QA**: A clinical decision-making dataset derived from the New England Journal of Medicine.

## Citation

If you use this code or framework, please cite:

```bibtex
@article{vacs2026,
  title={VACS: Value-Aligned Compositional Shielding for Multi-Agent Reasoning},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={2026}
}
```
