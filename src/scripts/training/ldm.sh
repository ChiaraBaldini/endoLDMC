### Stage 2: Training LDM ###
# This script is used to train the latent diffusion model on the endoscopic training set.
# We used a pretrained model from the HuggingFace repository: https://huggingface.co/stabilityai/stable-diffusion-2-1-base, and we fine-tuned it on our dataset.


seed=42
run_dir="LDM_txt_ft_v0"
stage1_uri=""
training_ids="project/outputs/ids_lar/train.tsv"
validation_ids="project/outputs/ids_lar/validation.tsv"
config_file="configs/ldm/ldm_v6.yaml"
scale_factor=0.1 
batch_size=4 
n_epochs=100
adv_start=10
eval_freq=1
num_workers=8 
experiment=${run_dir}
is_resumed=false #true
use_pretrained=1
clip_grad_norm_by=15.0 
clip_grad_norm_or_value='value'
img_width=640 #224
img_height=640 #224
source_model="stabilityai/stable-diffusion-2-1-base"  #"stabilityai/stable-diffusion-xl-base-1.0"  
torch_detect_anomaly=0 # whether to use torch.autograd.detect_anomaly() or not (0 not, 1 yes)


python3 src/python/training/train_ldm.py \
        --seed ${seed} \
        --run_dir ${run_dir} \
        --training_ids ${training_ids} \
        --validation_ids ${validation_ids} \
        --stage1_uri=${stage1_uri} \
        --config_file ${config_file} \
        --scale_factor=${scale_factor} \
        --batch_size ${batch_size} \
        --n_epochs ${n_epochs} \
        --eval_freq ${eval_freq} \
        --num_workers ${num_workers} \
        --experiment ${experiment} \
        --use_pretrained ${use_pretrained} \
        --source_model ${source_model} \
        --torch_detect_anomaly ${torch_detect_anomaly} \
        --img_width ${img_width} \
        --img_height ${img_height} \
        --clip_grad_norm_by ${clip_grad_norm_by} \
        --clip_grad_norm_or_value ${clip_grad_norm_or_value} \
        --"fine_tune"
        #--"use_default_report_text" \