import requests
import pandas as pd
import io
import os

def download_confirmed_cybersec():
    # URL confirmed via list_repo_files
    url = "https://huggingface.co/datasets/walledai/CyberSecEval/resolve/main/instruct/python-00000-of-00001.parquet"
    
    print(f"Downloading from {url}...")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            print("Download successful. Parsing Parquet...")
            try:
                df = pd.read_parquet(io.BytesIO(r.content))
            except ImportError:
                # Fallback if io.BytesIO fails with engine
                with open('temp_cyber.parquet', 'wb') as f:
                    f.write(r.content)
                df = pd.read_parquet('temp_cyber.parquet')
                
            print(f"Loaded {len(df)} rows.")
            # print(df.columns)
            # Columns likely: prompt, repo, file_path, line_text, etc. based on commit history
            
            # Map to our format
            data = []
            for _, row in df.iterrows():
                # Instruct dataset usually has 'prompt' or 'instruction'
                q = row.get('prompt', '') or row.get('instruction', '') or row.get('line_text', '')
                
                # In CyberSecEval, 'prompt' is the input.
                # The task is often to complete it securely or insecurely.
                # We want to test if VACS provides a SAFE answer.
                # So we treat 'prompt' as the question.
                # 'answer' column might not exist if it's just prompts for eval.
                # If so, we leave answer blank or use a placeholder for "Safe Refusal".
                
                if not q: continue
                
                # Filter for manageable length
                if len(q) > 1000: q = q[:1000] + "..."
                
                data.append({
                    'question': q,
                    'answer': "Safe Refusal", # Target answer for VACS
                    'options': "['Safe', 'Unsafe']",
                    'explanation': "This prompt triggers a cybersecurity risk check."
                })
            
            # Save top 100
            final_df = pd.DataFrame(data[:100])
            final_df.to_csv('data/cybersec_eval.csv', index=False)
            print(f"Saved {len(final_df)} real samples to data/cybersec_eval.csv")
            
        else:
            print(f"Failed: {r.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_confirmed_cybersec()
