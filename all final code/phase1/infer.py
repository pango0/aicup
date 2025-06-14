import os
import json
import torch
import random
import librosa
import zipfile
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset, Audio, load_dataset, Features, Value
from aicup import (DataCollatorSpeechSeq2SeqWithPadding,
      transcribe_with_timestamps,
      collate_batch_with_prompt_template,
      generate_annotated_audio_transcribe_parallel,OpenDeidBatchSampler)
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
)
from tqdm import tqdm

valid_dataset_list = []
t1_vaild_audio_folder = "/tmp2/zion/AICup/phase1/private_en_audio"
for file in sorted(os.listdir(t1_vaild_audio_folder)):
  if file.endswith(".wav"):
    try:
      file_path = os.path.join(t1_vaild_audio_folder, file)
      audio_array, sr = librosa.load(file_path, sr=16000)
      valid_dataset_list.append({"audio": {"path":file_path,'array':audio_array,'sampling_rate':sr},
                                 "sentence": ""})
    except Exception as e:
      print(e)
      print(f"Can't read {file_path}")

valid_dataset = Dataset.from_pandas(pd.DataFrame(valid_dataset_list))

# model_name = "/tmp2/77/aicup-slave-individual_prompting/phase1/whisper-large-v3-turbo-2/checkpoint-150"
# model = WhisperForConditionalGeneration.from_pretrained(model_name)
# processor = WhisperProcessor.from_pretrained(model_name)
# 這是你訓練儲存的 checkpoint
checkpoint_path = "whisper-large-v3-turbo-en/checkpoint-100"
# 這是 base model 名稱，用來載入 processor/tokenizer
base_model_name = "openai/whisper-large-v3-turbo"
# ✅ 載入模型（從 checkpoint）
model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
# ✅ 載入 processor（從 base model）
processor = WhisperProcessor.from_pretrained(base_model_name)
model.generation_config.language = 'en'
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

output_file = "task1_answer.txt"
json_output_file = "task1_answer_timestamps.json"
_mapping = {}

with open(output_file, "w", encoding="utf-8") as f:
  for _file in tqdm(valid_dataset):
    result = transcribe_with_timestamps(_file,model,processor)
    _mapping[_file['audio']['path'].split("/")[-1].split(".")[0]] = result
    f.write(f"{_file['audio']['path'].split('/')[-1].split('.')[0]}\t{result['text']}\n")

with open(json_output_file, "w", encoding="utf-8") as f:
  json.dump(_mapping, f, ensure_ascii=False)