import sys
import os
import random
import numpy as np

# Add parent directory to path to import vacs package
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from vacs.utils import load_data, simulate_agent_reasoning

def check_answer(predicted, ground_truth):
    """
    Checks if predicted answer matches ground truth.
    Handles cases like "(C) 10^-4" vs "C" or "(C)".
    """
    if not predicted:
        return False
    predicted = predicted.strip()
    ground_truth = ground_truth.strip()
    
    if predicted == ground_truth:
        return True
    
    # Check if one starts with the other (e.g. "A. Option" vs "A")
    if predicted.startswith(ground_truth) or ground_truth.startswith(predicted):
        return True
        
    # Check if ground truth is in predicted (e.g. "(C)" in "(C) 10^-4")
    if ground_truth in predicted:
        return True
        
    return False

def run_baselines(dataset_path: str, dataset_name: str, num_agents: int = 4):
    print(f"\nRunning Baselines on {dataset_name}...")
    
    data = load_data(dataset_path)
    if not data:
        print("No data found.")
        return

    # Baseline 1: Majority Vote
    majority_correct = 0
    # Baseline 2: Weighted Vote (simulated accuracy weights)
    weighted_correct = 0
    
    historical_accuracies = {i: 0.7 + np.random.normal(0, 0.05) for i in range(num_agents)}
    
    total_count = 0
    
    for case in data:
        question = case['question']
        options = case['options_list']
        ground_truth = case['answer']
        
        agent_answers = []
        for i in range(num_agents):
            # Use same simulation logic as VACS but without shielding
            # Simple simulation: 70% chance correct
            # But we need to pick a valid option string
            
            # Find the correct option string to simulate "correct" choice
            correct_option = None
            for opt in options:
                if check_answer(opt, ground_truth):
                    correct_option = opt
                    break
            
            if not correct_option and options:
                correct_option = options[0] # Fallback
                
            is_correct = random.random() < historical_accuracies[i]
            
            if is_correct and correct_option:
                ans = correct_option
            else:
                # Pick wrong option
                wrong_options = [o for o in options if o != correct_option]
                if not wrong_options:
                    wrong_options = options
                ans = random.choice(wrong_options)
                
            agent_answers.append(ans)
            
        # Majority Vote
        from collections import Counter
        counts = Counter(agent_answers)
        majority_ans = counts.most_common(1)[0][0]
        if check_answer(majority_ans, ground_truth):
            majority_correct += 1
            
        # Weighted Vote
        weighted_scores = {}
        for i, ans in enumerate(agent_answers):
            weighted_scores[ans] = weighted_scores.get(ans, 0) + historical_accuracies[i]
        
        if weighted_scores:
            weighted_ans = max(weighted_scores, key=weighted_scores.get)
            if check_answer(weighted_ans, ground_truth):
                weighted_correct += 1
            
        total_count += 1
        
    majority_acc = majority_correct / total_count if total_count > 0 else 0
    weighted_acc = weighted_correct / total_count if total_count > 0 else 0
    
    print(f"Results for {dataset_name}:")
    print(f"Majority Vote Accuracy: {majority_acc:.2%}")
    print(f"Weighted Vote Accuracy: {weighted_acc:.2%}")
    
    return majority_acc, weighted_acc

if __name__ == "__main__":
    run_baselines("data/gpqa_diamond.csv", "GPQA-Diamond")
    run_baselines("data/nejm_qa.csv", "NEJM-AI QA")
