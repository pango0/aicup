export HF_HOME=/tmp2/77

CUDA_VISIBLE_DEVICES=2 python train.py \
    --model_name_or_path yentinglin/Llama-3-Taiwan-8B-Instruct \
    --output_dir /tmp2/b10204022/aicup-slave-individual_prompting/tryFT/model \
    --dataset task2_answer_alpaca.json \
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