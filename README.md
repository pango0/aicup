# Environment Setup
Create conda env with `environment.yaml`

# Task 1
## Training

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
python src/split_training_set.py \
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
- `src/data_aug_en.py`: Generate English entries
- `src/data_aug_zh.py`: Generate Chinese entries
```bash
# e.x. for the LOCATION category, to generate English entries
python src/data_aug_en.py \
    --input_file train_split/train_LOCATIONS.json \
    --category "LOCATIONS" \
    --output_file train_split/extended_train_LOCATIONS.json
```
 
## Training
We use QLoRA fine-tuning to train a total of 8 models
- `unsloth/gemma-3-27b-it-bnb-4bit` is fine-tuned on all training data
- 7 * `Qwen/Qwen2.5-32B-Instruct` is each fine-tuned on the splitted training data and our augmented data
You can use the following command to train a model
```bash
# e.x. for training the CONTACT category
python src/train.py \
    --model_name_or_path "Qwen/Qwen2.5-32B-Instruct" \
    --output_dir "model/contact" \
    --dataset "train_split/extended_train_CONTACT.json" \
    --prompt_type "CONTACT_PROMPT" \
    --dataset_format alpaca \
    --bits 4 \
    --bf16 \
    --do_train \
    --max_steps 100 \
    --save_steps 100 \
    --per_device_train_batch_size 32 \
    --max_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type constant
```

## Prediction/Inference
### Split models
To inference the models trained on the splitted data, you should please modify the `config` in `src/inference_ft.py` to the correct model path and your desired output directory.
```bash
# e.x. for inferencing the CONTACT category
python src/inference_ft.py \
    -t "CONTACT" 
```
---
### Single model

---
### Merging results


## Utility files
`src/llm.py`: Simple wrapper around a Hugging Face–style transformer-based large language model (LLM) to streamline loading, inference, and prompt handling