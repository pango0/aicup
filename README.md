# Environment Setup
Create conda env with `environment.yaml`

# Task 1
We finetuned `openai/whisper-large-v3-turbo` model for speech recognition tasks in English and Chinese using the AICup dataset.
## Training
We finetuned two models for Chinese and English task. 
So, adjust the dataset path on either the English or Chinese dataset. 
This code also do inference with the last checkpoint. 

Train on English task:
```bash
python task1/train.py
```
Train on Chinese task:
```bash
python task1/train_zh.py
```
## Inference
Choose a particular checkpoint and run inference on  audio and generate transcriptions with timestamps.
```bash
python task1/infer.py
```
Ensure you update the following variables in infer.py:
```bash
checkpoint_path = "whisper-large-v3-turbo-en/checkpoint-100"
base_model_name = "openai/whisper-large-v3-turbo"
```
---
# Task 2
## Data Preprocessing
We define the following subcategories, where each category contains their related labels:

- NAME
- LOCATIONS
- ORGANIZATIONS
- DEMOGRAPHICS
- CONTACT
- IDENTIFIERS

Then split the data into different categories by running
```bash
python task2/split_training_set.py \
    --input_file train.json \
    --output_dir train_split
```
`train.json` should be in the following format
```json
[
  {
    "id": 19,
    "instruction": "Any overture of something that's kind of like a little white flag or peace offering to just get a week of peace, I'm not talking about permanent \"I'm going to placate and cow tow to you and to talk my needs in other...\" No. Just talking about lets...",
    "output": "[]"
  },
  {
    "id": 23,
    "instruction": "Yeah, I imagine it would — sorry, go ahead. So it's supposed to work immediately, right? Yep. So we'll see if I'm productive tomorrow. I hope I'm productive today. I've actually been trying to plan. If I do the titles today, then I can do my laundry tomorrow. Right. I probably could bring my computer and do titles while I'm doing my laundry. If I was — but I won't do that.",
    "output": "[]"
  },
  ...
]
```
Multiple `train_<category>.json` should be generated in `train_split`
## Data Augmentation
For each json file in `train_split/`, augment data if the number of entries < 300 (We generate 300 entries by default). 
- `task1/data_aug_en.py`: Generate English entries
- `task1/data_aug_zh.py`: Generate Chinese entries
```bash
# e.x. for the LOCATION category, to generate English entries
python task2/data_aug_en.py \
    --input_file train_split/train_LOCATIONS.json \
    --category "LOCATIONS" \
    --output_file train_split/extended_train_LOCATIONS.json
```
 
## Training
We use QLoRA fine-tuning to train a total of 8 models
- `unsloth/gemma-3-27b-it-bnb-4bit` is fine-tuned on all training data (`task2/train_whole.sh` is the script we used during our training)
- 7 * `Qwen/Qwen2.5-32B-Instruct` is each fine-tuned on the splitted training data and our augmented data (`task2/train_split.sh` is the script we used during our training)

## Prediction/Inference
### Split models
To inference the models trained on the splitted data, you should please modify the `config` in `task2/inference_split.py` to the correct model path and your desired output directory.
```bash
# e.x. for inferencing the CONTACT category
python task2/inference_split.py \
    -t "CONTACT" 
```
---
### Single model  
To inference the models trained on the whole data, you should please modify `task2/inference_whole.py` to the correct model path and your desired output directory.
```bash
python task2/inference_whole.py \
    --base_model_path (your base model)\
    --peft_path (checkpoint you want) \
```
---
### Merging results

Name the outputs of the two models `ans1.txt` and `ans2.txt`.

```bash
python task2/merge.py
```

It will merge the two files and output `merge.txt`.

## Utility files
`task2/llm.py`: Simple wrapper around a Hugging Face–style transformer-based large language model (LLM) to streamline loading, inference, and prompt handling
