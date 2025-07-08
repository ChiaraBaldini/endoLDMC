### Stage 1: Training AEKL ###
# This script is used to train the autoencoder model on the training set.
# In our case, this first stage was avoided, since we used a pretrained model from the HuggingFace repository: https://huggingface.co/stabilityai/stable-diffusion-2-1-base

seed=42
run_dir="aekl_v0"
training_ids="project/outputs/ids_lar/train.tsv"
validation_ids="project/outputs/ids_lar/validation.tsv"
config_file="configs/stage1/aekl_v0.yaml"
batch_size=2 
n_epochs=1000
adv_start=10
eval_freq=5
num_workers=4
experiment="AEKL_v0"

python3 src/python/training/train_aekl.py \
        --seed ${seed} \
        --run_dir ${run_dir} \
        --training_ids ${training_ids} \
        --validation_ids ${validation_ids} \
        --config_file ${config_file} \
        --batch_size ${batch_size} \
        --n_epochs ${n_epochs} \
        --adv_start ${adv_start} \
        --eval_freq ${eval_freq} \
        --num_workers ${num_workers} \
        --experiment ${experiment}\
        #   --"fine_tune" \
        #   --source_model ${source_model}        
