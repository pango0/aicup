# This script combines training and validation JSON files from specified directories.
import json
import os

# Define directories
train_dir = './extended_data_split'
val_dir = './val_split'
output_dir = './combined_data_split'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all training file names
train_files = [f for f in os.listdir(train_dir) if f.startswith("train_") and f.endswith(".json")]

for train_file in train_files:
    # Determine the corresponding validation file name
    suffix = train_file.replace("train_", "")
    val_file = f"val_{suffix}"
    
    train_path = os.path.join(train_dir, train_file)
    val_path = os.path.join(val_dir, val_file)
    output_path = os.path.join(output_dir, f"combined_{suffix}")
    
    # Read JSON contents
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # Combine (assumes list-style JSON data)
    if isinstance(train_data, list) and isinstance(val_data, list):
        combined = train_data + val_data
    elif isinstance(train_data, dict) and isinstance(val_data, dict):
        combined = {**train_data, **val_data}  # val_data overwrites train_data if keys overlap
    else:
        raise ValueError(f"Incompatible data types in {train_file} and {val_file}")

    # Save combined JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Combined {train_file} and {val_file} → {output_path}")