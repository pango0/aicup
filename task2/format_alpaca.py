import json
import pandas as pd

data = pd.read_csv("../phase1/task1_zh_answer.txt", sep='\t',header=None, names=["id", "text"])

alpaca_data = []
for _, row in data.iterrows():
    alpaca_data.append({
        "id": str(row["id"]),
        "instruction": row["text"]
    })

with open("task1_zh_alpaca.json", "w") as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)