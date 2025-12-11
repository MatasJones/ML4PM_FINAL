import pandas as pd
import os

# Mapping subset for Sarelli MG1
# signal_id for ball_valve_closed/open
# Based on previous mapping:
# Sarelli MG1:
# ball_valve_closed: 15454
# ball_valve_open: 15453
# (From the mapping dict in previous turn)

# Let's verify this from the full mapping if possible, but I'll use the IDs derived from the mapping I saw.
# Mapping snippet from user:
# 'signal_id': [..., 16769, 15454, 15453, 16679, 14156, 14154, ...]
# 'plant': [..., 'Sarelli', 'Sarelli', 'Sarelli', 'Sarelli', 'Sarelli', 'Sarelli', ...]
# 'machine_group': [..., 'MG1', 'MG1', 'MG1', 'MG1', 'MG1', 'MG1', ...]
# 'signal_name': [..., 'active_power', 'ball_valve_closed', 'ball_valve_open', ...]
#
# So 15454 is closed, 15453 is open.

file_path = '../axpo_challenge/data/Sarelli_MG1_testing_real_measurements.parquet'
print(f"Checking {file_path}")

if not os.path.exists(file_path):
    print("File not found.")
else:
    df = pd.read_parquet(file_path)
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns)
    
    # Check for specific signal IDs
    target_ids = [15454, 15453]
    subset = df[df['signal_id'].isin(target_ids)]
    print(f"Rows matching target IDs {target_ids}: {len(subset)}")
    
    if len(subset) > 0:
        print("Value counts for subset:")
        print(subset['signal_id'].value_counts())
        print(subset.head())
    else:
        print("No data found for these signal IDs in this file.")
        
    # Check what signal IDs ARE in the file
    print("Top 10 signal IDs in file:")
    print(df['signal_id'].value_counts().head(10))
