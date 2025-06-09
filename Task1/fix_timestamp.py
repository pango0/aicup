import json

from ftfy import fix_text

with open("task1_answer_timestamps.json", encoding="utf-8") as f:
    data = json.load(f)

for key, info in data.items():
    info["text"] = fix_text(info["text"])
    for seg in info["segments"]:
        seg["word"] = fix_text(seg["word"])

with open("task1_answer_timestamps_fixed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
