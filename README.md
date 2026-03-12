# VACS: Value-Aligned Compositional Shielding for Multi-Agent Reasoning

This repository contains the official implementation and experimental code for the paper **"VACS: Value-Aligned Compositional Shielding for Multi-Agent Reasoning"**.

VACS is a four-layer framework designed to align multi-agent reasoning systems with heterogeneous value priorities (e.g., safety vs. efficiency). It integrates:
1.  **Value Profiling**: Inferring latent agent value weights from behavioral preferences (Layer 1).
2.  **Compositional Shielding**: Enforcing safety constraints via assume-guarantee reasoning (Layer 2).
3.  **Hamiltonian Consensus**: Resolving conflicts using nucleolus-weighted optimization (Layer 3).
4.  **Faithful Explanation**: Generating explanations grounded in the formal proof structure (Layer 4).

## 📂 Repository Structure

```text
VACS_MultiAgentReasoning/
├── data/                           # Benchmark datasets (CSV format)
│   ├── math_subset.csv             # MathInstruct (Reasoning) subset (N=200)
│   ├── nejm_qa.csv                 # NEJM-AI QA (Clinical) benchmark
│   └── cybersec_eval.csv           # CyberSec-Eval (Incident Response) benchmark
│
├── experiments/                    # Experiment reproduction scripts
│   ├── comprehensive_experiments.py # [Main] Runs Table 3 (Accuracy/LIR) & Table 4 (Scalability)
│   ├── explanation_metric.py       # [Main] Runs Table 7 (Faithfulness Validation)
│   ├── ablation_study.py           # Runs ablation analysis
│   ├── run_vacs.py                 # Core runner for VACS pipeline
│   └── run_baselines.py            # Runner for baseline methods
│
├── vacs/                           # Core Framework Implementation
│   ├── layer1_value_profiling.py   # Layer 1: Bradley-Terry & MaxEnt IRL
│   ├── layer2_shielding.py         # Layer 2: Lean-DSL Shielding
│   ├── layer3_consensus.py         # Layer 3: Nucleolus & Hamiltonian Optimization
│   ├── layer4_explanation.py       # Layer 4: Critical Path Extraction
│   ├── download_datasets.py        # Utility to fetch real datasets (CyberSecEval)
│   └── utils.py                    # Helper functions (loading, simulation)
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🛠️ Setup & Requirements

The framework is implemented in Python (3.10+ recommended).

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Key libraries: `numpy`, `pandas`, `scipy`, `tqdm`, `requests`, `huggingface_hub`.*

2.  **Data Preparation**:
    The processed datasets are already included in the `data/` folder.
    *   If you need to re-download the real **CyberSec-Eval** dataset from HuggingFace, run:
        ```bash
        python vacs/download_datasets.py
        ```

## 🚀 Reproducing Results

### 1. Main Performance (Table 3) & Scalability (Table 4)
To reproduce the accuracy and logical inconsistency rate (LIR) results for VACS and all baselines (Weighted Vote, Shield-Only, etc.) across the three benchmarks:

```bash
python experiments/comprehensive_experiments.py
```
*   **Output**: Prints accuracy tables and LIR metrics to the console.
*   **Note**: This script simulates the full pipeline including baseline comparisons.

### 2. Explanation Faithfulness (Table 7)
To validate the faithfulness of VACS explanations against baselines (Gradient Saliency, Verifier Trace) using metrics like Sufficiency, Necessity, and Counterfactual Drop:

```bash
python experiments/explanation_metric.py
```
*   **Output**: Generates a summary table of faithfulness metrics and saves detailed results to `experiments/explanation_results.csv`.

### 3. Ablation Study
To analyze the contribution of each VACS layer (removing Layer 1, Layer 2, etc.):

```bash
python experiments/ablation_study.py
```

## 📊 Datasets

The repository uses three primary benchmarks:
1.  **MathInstruct-Subset**: A challenging subset of mathematical reasoning tasks requiring rigorous step-by-step logic.
2.  **NEJM-AI QA**: A clinical decision-making dataset derived from New England Journal of Medicine case challenges.
3.  **CyberSec-Eval**: A cybersecurity incident response benchmark (derived from WalledAI's instruct dataset) testing safety-critical decision making.

## 🔗 Citation

If you use this code or framework, please cite:

```bibtex
@article{vacs2026,
  title={VACS: Value-Aligned Compositional Shielding for Multi-Agent Reasoning},
  author={Zhang, Yiyao and Goel, Diksha and Ahmad, Hussain and Shen, Jun},
  journal={Under Review},
  year={2026}
}
```
