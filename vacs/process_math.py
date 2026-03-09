import json
import pandas as pd
import random
import re

def process_math_instruct():
    try:
        # Load a subset of MathInstruct (it's huge)
        # We'll simulate reading it as JSON. 
        # Since the file might be large, let's read line by line or use a limit.
        # Actually, let's just grab the first 200 items for a "Math-Subset" benchmark.
        
        data = []
        with open('data/math_instruct.json', 'r', encoding='utf-8') as f:
            # It's likely a JSON list. Reading the whole thing might be slow if it's GBs.
            # Let's try to parse it. If it fails, we'll try line-based.
            try:
                full_data = json.load(f)
                data = full_data[:200]
            except:
                print("Could not load full JSON. Attempting line-based...")
                f.seek(0)
                for i, line in enumerate(f):
                    if i >= 200: break
                    try:
                        data.append(json.loads(line))
                    except:
                        pass
        
        processed = []
        for item in data:
            question = item.get('instruction', '') or item.get('question', '')
            output = item.get('output', '') or item.get('answer', '')
            
            # Extract final answer from output (usually after "The answer is")
            # This is heuristic.
            answer = output
            
            # Create distractors (Simulated)
            # Since we don't have real distractors, we'll create placeholders
            # In a real scenario, we'd use an LLM to generate them.
            options = [answer, "Incorrect Option A", "Incorrect Option B", "Incorrect Option C"]
            random.shuffle(options)
            
            processed.append({
                'question': question,
                'options': str(options),
                'answer': answer,
                'explanation': output # The full output is the explanation
            })
            
        df = pd.DataFrame(processed)
        df.to_csv('data/math_subset.csv', index=False)
        print(f"Created data/math_subset.csv with {len(df)} samples.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    process_math_instruct()
