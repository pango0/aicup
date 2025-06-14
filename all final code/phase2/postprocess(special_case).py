import json
import pandas as pd

answer = pd.read_csv("private_match_LLM_zh.txt", sep="\t", header=None, names=["id", "category", "start", "end", "text"])

with open("private_match_LLM_zh_filtered.txt", "w", encoding="utf-8") as f:
    for i in range(len(answer)):
        if answer["start"][i] == answer["end"][i]:
            print(f"Error in line {i}: start and end are the same ({answer['start'][i]})")
            continue
        if answer["end"][i] - answer["start"][i] > 20:
            print(f"Error in line {i}: start {answer['start'][i]}, end {answer['end'][i]}")
            continue
        f.write(f"{answer['id'][i]}\t{answer['category'][i]}\t{answer['start'][i]}\t{answer['end'][i]}\t{answer['text'][i]}\n")