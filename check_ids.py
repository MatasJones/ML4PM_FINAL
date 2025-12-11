import pandas as pd
import os

file_path = '../axpo_challenge/data/Sarelli_MG1_testing_real_measurements.parquet'
print(f"Reading {file_path}")

try:
    df = pd.read_parquet(file_path)
    print("Columns:", df.columns)
    
    if 'signal_id' in df.columns:
        counts = df['signal_id'].value_counts()
        print("Signal IDs present:")
        print(counts)
        
        # Check against expected IDs
        # Sarelli MG1 expected: 15454 (closed), 15453 (open)
        expected = [15454, 15453]
        for eid in expected:
            if eid in counts.index:
                print(f"  ID {eid} FOUND. Count: {counts[eid]}")
            else:
                print(f"  ID {eid} NOT FOUND.")
    else:
        print("Column 'signal_id' not found.")
        
except Exception as e:
    print(f"Error: {e}")
