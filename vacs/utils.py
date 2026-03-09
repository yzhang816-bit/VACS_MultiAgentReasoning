import csv
from typing import List, Dict
import ast
import re

def load_data(filepath: str) -> List[Dict[str, str]]:
    """
    Loads data from a CSV file.
    Assumes header: question, options, answer, explanation
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'options' not in row:
                    # Maybe it's 'Choices' or something else if csv structure changed
                    continue
                    
                options_str = row['options'].strip()
                row['options_list'] = []
                
                if options_str.startswith("[") and options_str.endswith("]"):
                    try:
                        row['options_list'] = ast.literal_eval(options_str)
                    except Exception as e:
                        print(f"Error parsing list options: {e}")
                        row['options_list'] = [opt.strip(" '\"[]") for opt in options_str.split(',')]
                else:
                    # Handle multiline or "A. ... B. ..." format
                    if '\n' in options_str:
                        # Split by newline
                        row['options_list'] = [opt.strip() for opt in options_str.split('\n') if opt.strip()]
                    elif re.search(r'[A-D]\.', options_str):
                         # Split by letter patterns like "A. ", "B. "
                         # This is a bit tricky with regex split, keeping delimiters
                         # Simpler approach: if no newline, maybe just treat as one block or try to split
                         # For now, let's assume if it's not a list and no newlines, it might be a single string
                         row['options_list'] = [options_str]
                    else:
                         row['options_list'] = [opt.strip() for opt in options_str.split(',')]
                
                # Filter out empty options
                row['options_list'] = [o for o in row['options_list'] if o]
                
                data.append(row)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []
    except Exception as e:
        print(f"Error loading data: {e}")
        return []
    return data

def simulate_agent_reasoning(question: str, options: List[str], correct_answer: str, agent_profile: dict, accuracy_boost: float = 0.0, llm_response: dict = None) -> Dict:
    """
    Simulates an agent's reasoning process.
    If llm_response is provided, uses the pre-computed LLM answer/explanation.
    Otherwise, simulates based on profile.
    """
    import random
    from vacs.layer2_shielding import ReasoningStep
    
    # Check if we have a real LLM response for this agent
    if llm_response and llm_response.get('answer'):
        final_answer = llm_response['answer']
        explanation_text = llm_response.get('explanation', '')
        
        # Determine correctness based on the LLM's answer vs ground truth
        # Note: 'is_correct' flag in return dict is for evaluation
        # We need to check if final_answer matches correct_answer
        # Simple check logic (similar to check_answer in run_vacs)
        is_correct = False
        
        # Robust normalization for correctness check inside simulation
        def normalize_text(text):
            t = str(text).lower()
            # Remove common latex/markdown chars
            for char in ['\\', '{', '}', '(', ')', '[', ']', '$', '*', '`', '"', "'", ' ']:
                t = t.replace(char, '')
            t = t.replace('times', '').replace('cdot', '')
            return t
            
        # 2. Extract answer from Boxed if present
        # Many math models output \boxed{answer}
        boxed_match = re.search(r'\\boxed{([^}]+)}', final_answer)
        if boxed_match:
            final_answer = boxed_match.group(1)
            
        norm_ans = normalize_text(final_answer)
        norm_gt = normalize_text(correct_answer)
        
        if norm_ans == norm_gt:
            is_correct = True
        elif norm_gt in norm_ans:
            is_correct = True
        elif len(norm_gt) > 3 and norm_ans in norm_gt:
            is_correct = True
            
        # Build trajectory from explanation
        trajectory = []
        trajectory.append(ReasoningStep(
            content=f"Analyzing question: {question[:50]}...",
            action_type="analyze"
        ))
        
        # Split explanation into chunks to simulate steps
        # Simple heuristic: split by newlines or periods
        steps = [s.strip() for s in explanation_text.split('\n') if s.strip()]
        if not steps:
            steps = [explanation_text]
            
        for i, step_content in enumerate(steps[:5]): # Limit to 5 steps
            trajectory.append(ReasoningStep(
                content=step_content[:200], # Truncate for brevity
                action_type="deduce",
                premises=[i]
            ))
            
        trajectory.append(ReasoningStep(
            content=f"Therefore, the answer is {final_answer}.",
            action_type="conclude",
            premises=[len(trajectory)-1]
        ))
        
        return {
            "trajectory": trajectory,
            "answer": final_answer,
            "is_correct": is_correct
        }

    # Fallback to simulation if no LLM response
    # Extract profile tendencies
    completeness_weight = agent_profile.get("Logical Completeness", 0.2)
    conciseness_weight = agent_profile.get("Conciseness", 0.2)
    soundness_weight = agent_profile.get("Logical Soundness", 0.2)
    safety_weight = agent_profile.get("Safety", 0.2)
    
    # Determine correctness probability based on profile (Rigorous/Sound agents are more accurate)
    # Base accuracy for hard tasks (GPQA/NEJM) is lower. Random is 0.25.
    # Experts might reach 0.7.
    # Default profile (0.2 weights) -> 0.3 + 0.4*(0.5) = 0.5.
    # Good profile (0.8 weights) -> 0.3 + 0.4*(2.0) = 1.1 (capped).
    accuracy_prob = 0.05 + 0.45 * (completeness_weight + soundness_weight + 0.8 * safety_weight) + accuracy_boost
    
    # Cap probability
    accuracy_prob = min(0.98, accuracy_prob)
    
    # Ensure options is not empty
    if not options:
        options = ["A", "B", "C", "D"] # Fallback
        
    is_correct = random.random() < accuracy_prob
    
    # Check if correct_answer is in options. If not (e.g. format mismatch), just pick random or matching letter
    # NEJM answer is 'D', options are 'D. You should...'
    # We need to match letter.
    
    matched_option = None
    for opt in options:
        if correct_answer in opt or (len(correct_answer) == 1 and opt.startswith(correct_answer)):
            matched_option = opt
            break
    
    if matched_option and is_correct:
        final_answer = matched_option
    else:
        # Pick a wrong option
        wrong_options = [o for o in options if o != matched_option]
        if not wrong_options:
             wrong_options = options 
        
        # Correlated Error Simulation:
        # Weak agents tend to fall for the same distractor (e.g., the first wrong option)
        # 80% chance to pick the first available wrong option (mimicking a "popular" distractor)
        if len(wrong_options) > 0 and random.random() < 0.8:
            final_answer = wrong_options[0]
        else:
            final_answer = random.choice(wrong_options)
    
    # If we couldn't match, just return something
    if not final_answer:
        final_answer = options[0]

    trajectory = []
    
    # Step 1: Analyze question
    trajectory.append(ReasoningStep(
        content=f"Analyzing the question: {question[:50]}...",
        action_type="analyze"
    ))
    
    # Step 2: Recall knowledge (more detailed if completeness is high)
    if completeness_weight > 0.3:
        trajectory.append(ReasoningStep(
            content=f"Recalling relevant principles...",
            action_type="recall",
            premises=[0]
        ))
        trajectory.append(ReasoningStep(
            content=f"Verifying assumptions...",
            action_type="verify",
            premises=[1]
        ))
    else:
        trajectory.append(ReasoningStep(
            content="Recalling key facts.",
            action_type="recall",
            premises=[0]
        ))
        
    # Step 3: Deduction
    trajectory.append(ReasoningStep(
        content=f"Deducing that {final_answer[:20]}... is likely.",
        action_type="deduce",
        premises=[len(trajectory)-1]
    ))
    
    # Step 4: Final conclusion
    trajectory.append(ReasoningStep(
        content=f"Therefore, the answer is {final_answer[:50]}.",
        action_type="conclude",
        premises=[len(trajectory)-1]
    ))
    
    return {
        "trajectory": trajectory,
        "answer": final_answer,
        "is_correct": is_correct
    }
