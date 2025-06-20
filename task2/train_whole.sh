CUDA_VISIBLE_DEVICES=3,4 
python train.py \
    --model_name_or_path unsloth/gemma-3-27b-it-bnb-4bit \
    --output_dir model \
    --dataset train_alpaca.json \
    --dataset_format alpaca \
    --bits 4 \
    --bf16 \
    --do_train \
    --max_steps 200 \
    --save_steps 20 \
    --per_device_train_batch_size 4 \
    --max_train_epochs 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type constant