# This script processes two text files, task1.txt and task2.txt, to create a JSON file in a specific format for training for phase 2.
import json
from collections import defaultdict

# Step 1: Load task1.txt
with open("task1.txt", "r", encoding="utf-8") as f:
     id_to_text = {}
     for line in f:
          parts = line.strip().split(maxsplit=1)
          if len(parts) == 2:
               utt_id, text = parts
               id_to_text[int(utt_id)] = text

# Step 2: Load task2.txt
entities = defaultdict(list)
with open("task2.txt", "r", encoding="utf-8") as f:
     for line_num, line in enumerate(f, 1):
          parts = line.strip().split('\t')
          if len(parts) != 5:
               print(f"⚠️ Line {line_num} malformed (expected 5 parts, got {len(parts)}): {line.strip()}")
               continue  # skip this line
          utt_id_str, label, start, end, text = parts
          utt_id = int(utt_id_str)
          entities[utt_id].append({label: text})

# Step 3: Combine into target format
results = []
for utt_id, instruction in id_to_text.items():
     output = entities.get(utt_id, [])
     results.append({
          "id": utt_id,
          "instruction": instruction,
          "output": json.dumps(output, ensure_ascii=False)
     })

# Step 4: Save to train.json
with open("train.json", "w", encoding="utf-8") as out_file:
     json.dump(results, out_file, indent=2, ensure_ascii=False)