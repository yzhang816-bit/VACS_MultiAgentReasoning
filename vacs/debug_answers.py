import pandas as pd
import re

def normalize(text):
    text = str(text).strip()
    # Remove Latex wrappers
    text = text.replace(r'\(', '').replace(r'\)', '').replace(r'\[', '').replace(r'\]', '')
    text = text.replace(r'\text{', '').replace(r'}', '')
    text = text.replace(r'\,', ' ').replace(r'\ ', ' ')
    # Remove markdown bold/italic
    text = text.replace('**', '').replace('*', '')
    # Normalize spaces
    text = " ".join(text.split())
    return text

def debug_answers():
    df = pd.read_csv('data/gpqa_diamond_full.csv')
    
    print(f"Total rows: {len(df)}")
    
    for idx, row in df.iterrows():
        gt = row['answer']
        pred = row.get('deepseek_ans', '') # DeepSeek is usually good
        
        norm_gt = normalize(gt)
        norm_pred = normalize(pred)
        
        match = False
        if norm_pred == norm_gt:
            match = True
        elif norm_gt in norm_pred:
            match = True
        elif len(norm_gt) > 2 and norm_pred in norm_gt:
            match = True
            
        if not match and idx < 10:
            print(f"\nRow {idx} FAIL:")
            print(f"GT:   '{gt}' -> '{norm_gt}'")
            print(f"Pred: '{pred}' -> '{norm_pred}'")

if __name__ == "__main__":
    debug_answers()
