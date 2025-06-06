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

def set_torch_seed(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benckmark = False
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_torch_seed()

# TODO: Change the paths to your audio dataset and transcription files
t1_train_audio_folder = "audio_dataset/TRAINGING_DATASET_1/Training_Dataset_01/audio"
t1_train_transcription_file = "audio_dataset/TRAINGING_DATASET_1/Training_Dataset_01/task1_answer.txt"
t2_train_audio_folder = "/tmp2/zion/AICup/training_dataset_2/audio"
t2_train_transcription_file = "/tmp2/zion/AICup/training_dataset_2/task1_answer.txt"
transcripts, dataset_list = {}, []

with open(t1_train_transcription_file, "r", encoding="utf-8") as f:
  for line in f:
    if line.strip():
      parts = line.strip().split("\t", 1)
      if len(parts) == 2:
        filename, transcript = parts
        transcripts[filename] = transcript

with open(t2_train_transcription_file, "r", encoding="utf-8") as f:
  for line in f:
    if line.strip():
      parts = line.strip().split("\t", 1)
      if len(parts) == 2:
        filename, transcript = parts
        transcripts[filename] = transcript

for file in sorted(os.listdir(t1_train_audio_folder)):
  if file.endswith(".wav") and file.split(".")[0] in transcripts:
    try:
      file_path = os.path.join(t1_train_audio_folder, file)
      audio_array, sr = librosa.load(file_path, sr=16000)
      dataset_list.append({"audio":
                 {
                  "path":file_path,
                  "array":audio_array,
                  "sampling_rate":sr
                 },
                 "sentence": transcripts[file.split(".")[0]]})
    except Exception as e:
      print(e)
      print(f"Can't read {file_path}:{e}")
      
for file in sorted(os.listdir(t2_train_audio_folder)):
  if file.endswith(".wav") and file.split(".")[0] in transcripts:
    try:
      file_path = os.path.join(t2_train_audio_folder, file)
      audio_array, sr = librosa.load(file_path, sr=16000)
      dataset_list.append({"audio":
                 {
                  "path":file_path,
                  "array":audio_array,
                  "sampling_rate":sr
                 },
                 "sentence": transcripts[file.split(".")[0]]})
    except Exception as e:
      print(e)
      print(f"Can't read {file_path}:{e}")

dataset = Dataset.from_pandas(pd.DataFrame(dataset_list))

print("audio sample:",len(dataset[0]['audio']['array']))
print("audio duration:",len(dataset[0]['audio']['array'])/16000)


# TODO: Change the split ratio as needed
split_ratio = 0.8
train_size = int(len(dataset) * split_ratio)
dataset = dataset.train_test_split(train_size=train_size,
     test_size=len(dataset) - train_size, shuffle=True, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

print(f"Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")

# TODO: Change the model name to the desired Whisper model
model_name = "openai/whisper-large-v3-turbo"  #"small", "medium", "large"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left'
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.generation_config.language = None
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

from jiwer import mer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from tqdm import tqdm

def calculate_mer(ground_truth_texts, predicted_texts):
  """ Mix Error Rate (MER) English only"""
  mer_scores = {}
  total_mer = 0
  count = 0

  normalizer = BasicTextNormalizer()

  for filename, ref_text in ground_truth_texts.items():
    if filename in predicted_texts:
      pred_text = predicted_texts[filename]
      ref_text = normalizer(ref_text)
      pred_text = normalizer(pred_text)
      mer_score = mer(ref_text, pred_text)
      mer_scores[filename] = mer_score
      total_mer += mer_score
    else:
      mer_scores[filename] = 1
      total_mer += 1
    count += 1

  average_mer = total_mer / count if count != 0 else 0
  return mer_scores, average_mer

def evaluate_mer(model, processor, dataset):
  predictions = {}
  references = {}

  for sample in tqdm(dataset, desc="Transcribing"):
    filename = sample["audio"]["path"].split("/")[-1].split(".")[0]
    audio_array = sample['audio']['array']
    sr = sample['audio']['sampling_rate']
    input_features = processor.feature_extractor(
        audio_array,
        sampling_rate=sr,
        return_tensors="pt"
    ).input_features.to(model.device)
    with torch.no_grad():
      predicted_ids = model.generate(input_features)
      transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    predictions[filename] = transcription
    references[filename] = sample["sentence"].strip()

  mer_scores, avg_mer = calculate_mer(references, predictions)
  return mer_scores, avg_mer, predictions, references

def preprocess_dataset(batch):

  audio = batch["audio"]
  batch["input_features"] = processor.feature_extractor(audio["array"],
    sampling_rate=audio["sampling_rate"]).input_features[0]
  batch["labels"] = tokenizer(batch["sentence"]).input_ids

  return batch

train_dataset = train_dataset.map(preprocess_dataset, remove_columns=dataset.column_names["train"]) #,num_proc=4
test_dataset = test_dataset.map(preprocess_dataset, remove_columns=dataset.column_names["test"])

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

d = [test_dataset[i] for i in range(2)]
batch = data_collator(d)

from jiwer import mer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


normalizer = BasicTextNormalizer()

def mixed_tokenizer(text):
  text = normalizer(text.strip())
  tokens = []
  temp_token = ""
  # print(text)
  for char in text:
    if '\u4e00' <= char <= '\u9fff':
      if temp_token:
          tokens.append(temp_token)
          temp_token = ""
      tokens.append(char)
    elif char.isspace():
      if temp_token:
        tokens.append(temp_token)
        temp_token = ""
    else:
      temp_token += char
  if temp_token:
    tokens.append(temp_token)
  return tokens

def compute_metrics(eval_pred):
  predictions, labels = eval_pred

  decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
  decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

  decoded_preds = [normalizer(pred.strip()) for pred in decoded_preds]
  decoded_labels = [normalizer(label.strip()) for label in decoded_labels]

  paired = [
      (ref, hyp) for ref, hyp in zip(decoded_labels, decoded_preds)
      if ref.strip() != "" and hyp.strip() != ""
  ]

  filtered_labels, filtered_preds = zip(*paired) if paired else ([], [])
  if len(filtered_labels) == 0:
    return {"mer": 1.0}

  ref_tokens = [mixed_tokenizer(t) for t in filtered_labels]
  pred_tokens = [mixed_tokenizer(t) for t in filtered_preds]
  # print(ref_tokens)
  # print(pred_tokens)
  ref_strs = [" ".join(tokens) for tokens in ref_tokens]
  pred_strs = [" ".join(tokens) for tokens in pred_tokens]

  try:
    score = mer(ref_strs, pred_strs)
  except Exception as e:
    print("Error during MER computation:", e)
    score = 1.0

  return {"mer": score}

# TODO: Adjust the training arguments as needed
training_args = Seq2SeqTrainingArguments(
  output_dir="whisper-large-v3-turbo-2",
  report_to="none",
  num_train_epochs=3,
  per_device_train_batch_size=2,
  per_device_eval_batch_size=2,
  evaluation_strategy="steps",
  eval_steps=30,
  save_strategy="steps",
  save_steps=30,
  predict_with_generate=True,
  logging_dir="whisper-large-v3-turbo-2/logs",
  logging_steps=5,
  fp16=True,
  learning_rate=1e-5,
  warmup_ratio=0.1,
  gradient_accumulation_steps=4,
  dataloader_num_workers=1,
)

trainer = Seq2SeqTrainer(
  model=model,
  args=training_args,
  train_dataset=train_dataset,
  eval_dataset=test_dataset,
  tokenizer=processor.tokenizer,
  data_collator=data_collator,
  compute_metrics=compute_metrics
)

trainer.train()

# TODO: Change the save directory as needed
save_directory = "whisper-large-v3-turbo-2"

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
processor.save_pretrained(save_directory)

print(f"Model saved to {save_directory}")

valid_dataset_list = []
t1_vaild_audio_folder = "/tmp2/77/aicup-slave-individual_prompting/phase1/audio_dataset/TRAINGING_DATASET_1/Validation_Dataset/audio"
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