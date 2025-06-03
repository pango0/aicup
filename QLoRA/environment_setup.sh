conda create -n qlora python=3.10 -y
conda activate qlora

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install transformers datasets accelerate peft trl bitsandbytes
