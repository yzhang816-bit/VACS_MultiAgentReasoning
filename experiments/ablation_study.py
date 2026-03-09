import sys
import os

# Add parent directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from experiments.run_vacs import run_vacs_experiment

def run_ablation():
    print("Starting Ablation Study on GPQA-Diamond (Full)...")
    
    dataset = "data/gpqa_diamond_full.csv"
    name = "GPQA-Diamond"
    
    results = {}
    
    # 1. Full VACS
    print("\n--- Full VACS ---")
    results['Full'] = run_vacs_experiment(dataset, name, ablation="quiet")
    
    # 2. w/o Layer 1
    print("\n--- w/o Layer 1 (Uniform Value Profiles) ---")
    results['No L1'] = run_vacs_experiment(dataset, name, ablation="no_layer1")
    
    # 3. w/o Layer 2
    print("\n--- w/o Layer 2 (No Shields) ---")
    results['No L2'] = run_vacs_experiment(dataset, name, ablation="no_layer2")
    
    # 4. w/o Layer 3
    print("\n--- w/o Layer 3 (Majority Vote Consensus) ---")
    results['No L3'] = run_vacs_experiment(dataset, name, ablation="no_layer3")
    
    # 5. w/o Layer 4
    print("\n--- w/o Layer 4 (No Explanation) ---")
    results['No L4'] = run_vacs_experiment(dataset, name, ablation="no_layer4")
    
    print("\n=== Ablation Results Summary (GPQA-Diamond N=198) ===")
    print(f"{'Configuration':<20} | {'Acc':<8} | {'LIR':<8} | {'Faith':<8}")
    print("-" * 50)
    for config, (acc, lir, faith) in results.items():
        print(f"{config:<20} | {acc:.2%} | {lir:.2%} | {faith:.2%}")
        
    # Also run on NEJM for confirmation
    print("\n=== Running on NEJM-AI QA (N=857) ===")
    dataset_nejm = "data/nejm_qa.csv"
    
    print("\n--- Full VACS (NEJM) ---")
    acc_full, lir_full, _ = run_vacs_experiment(dataset_nejm, "NEJM", ablation="quiet")
    
    print("\n--- Majority Vote (NEJM) ---")
    acc_mv, lir_mv, _ = run_vacs_experiment(dataset_nejm, "NEJM", ablation="no_layer3")
    
    print(f"\nNEJM Comparison: VACS {acc_full:.2%} vs Majority Vote {acc_mv:.2%}")
    
    # Also run on MathInstruct (Subset)
    print("\n=== Running on MathInstruct (Subset N=200) ===")
    dataset_math = "data/math_subset.csv"
    
    print("\n--- Full VACS (Math) ---")
    acc_math_full, lir_math_full, _ = run_vacs_experiment(dataset_math, "MathInstruct-Subset", ablation="quiet")
    
    print("\n--- Majority Vote (Math) ---")
    acc_math_mv, lir_math_mv, _ = run_vacs_experiment(dataset_math, "MathInstruct-Subset", ablation="no_layer3")
    
    print(f"\nMath Comparison: VACS {acc_math_full:.2%} vs Majority Vote {acc_math_mv:.2%}")

if __name__ == "__main__":
    run_ablation()
