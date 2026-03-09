import pandas as pd

def check_metadata():
    try:
        df = pd.read_parquet('data/gpqa_diamond_full.parquet')
        print("Metadata type:", type(df.iloc[0]['metadata']))
        print("Metadata content:", df.iloc[0]['metadata'])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_metadata()
