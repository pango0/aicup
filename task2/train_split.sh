export HF_HOME=/work/b11902044

MODEL_ID="Qwen/Qwen2.5-32B-Instruct"
DATASET="/work/b11902044/aicup/combined_data_split"
BATCH_SIZE=32
MAX_STEP=100

# GPU 0: CONTACT then LOCATIONS
(
  export CUDA_VISIBLE_DEVICES=0

  python train.py \
    --model_name_or_path $MODEL_ID \
    --output_dir /work/b11902044/aicup/model/contact \
    --dataset $DATASET/combined_CONTACT.json \
    --prompt_type "CONTACT_PROMPT" \
    --dataset_format alpaca \
    --bits 4 \
    --bf16 \
    --do_train \
    --max_steps $MAX_STEP \
    --save_steps 100 \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type constant

  python train.py \
    --model_name_or_path $MODEL_ID \
    --output_dir /work/b11902044/aicup/model/locations \
    --dataset $DATASET/combined_LOCATIONS.json \
    --prompt_type "LOCATIONS_PROMPT" \
    --dataset_format alpaca \
    --bits 4 \
    --bf16 \
    --do_train \
    --max_steps $MAX_STEP \
    --save_steps 100 \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type constant
) &

# GPU 1: DATE_TIME then NAMES
(
  export CUDA_VISIBLE_DEVICES=1

  python train.py \
    --model_name_or_path $MODEL_ID \
    --output_dir /work/b11902044/aicup/model/date_time \
    --dataset $DATASET/combined_DATE_TIME.json \
    --prompt_type "DATE_TIME_PROMPT" \
    --dataset_format alpaca \
    --bits 4 \
    --bf16 \
    --do_train \
    --max_steps $MAX_STEP \
    --save_steps 100 \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type constant

  python train.py \
    --model_name_or_path $MODEL_ID \
    --output_dir /work/b11902044/aicup/model/names \
    --dataset $DATASET/combined_NAMES.json \
    --prompt_type "NAMES_PROMPT" \
    --dataset_format alpaca \
    --bits 4 \
    --bf16 \
    --do_train \
    --max_steps $MAX_STEP \
    --save_steps 100 \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type constant
) &

# GPU 2: DEMOGRAPHICS then ORGANIZATIONS
(
  export CUDA_VISIBLE_DEVICES=2

  python train.py \
    --model_name_or_path $MODEL_ID \
    --output_dir /work/b11902044/aicup/model/demographics \
    --dataset $DATASET/combined_DEMOGRAPHICS.json \
    --prompt_type "DEMOGRAPHICS_PROMPT" \
    --dataset_format alpaca \
    --bits 4 \
    --bf16 \
    --do_train \
    --max_steps $MAX_STEP \
    --save_steps 100 \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type constant

  python train.py \
    --model_name_or_path $MODEL_ID \
    --output_dir /work/b11902044/aicup/model/organizations \
    --dataset $DATASET/combined_ORGANIZATIONS.json \
    --prompt_type "ORGANIZATIONS_PROMPT" \
    --dataset_format alpaca \
    --bits 4 \
    --bf16 \
    --do_train \
    --max_steps $MAX_STEP \
    --save_steps 100 \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type constant
) &

# GPU 3: IDENTIFIERS only
(
  export CUDA_VISIBLE_DEVICES=3

  python train.py \
    --model_name_or_path $MODEL_ID \
    --output_dir /work/b11902044/aicup/model/identifiers \
    --dataset $DATASET/combined_IDENTIFIERS.json \
    --prompt_type "IDENTIFIERS_PROMPT" \
    --dataset_format alpaca \
    --bits 4 \
    --bf16 \
    --do_train \
    --max_steps $MAX_STEP \
    --save_steps 100 \
    --per_device_train_batch_size $BATCH_SIZE \
    --max_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type constant
) &

wait