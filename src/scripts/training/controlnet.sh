### Stage 3: Training ControlNet module ###
# This script is used to train the ControlNet module (integrated into the frozen latent diffusion model) on the endoscopic training set.
# We used a pretrained model from the HuggingFace repository: https://huggingface.co/stabilityai/stable-diffusion-2-1-base, and we fine-tuned it on our dataset.


seed=42
run_dir="controlnet_txt_bb_ft_v0"
ddpm_uri="project/outputs/runs/LDM_txt_fn_v0/diffusion_best_model.pth" # ddpm_uri is the path to the previously fine-tuned diffusion model, which is used to initialize the ControlNet
stage1_uri=""
training_ids="project/outputs/ids_lar/train.tsv"
validation_ids="project/outputs/ids_lar/validation.tsv"
config_file="configs/controlnet/controlnet_v1.yaml"
scale_factor=0.1
batch_size=4 
n_epochs=100
eval_freq=1
num_workers=10 
experiment=${run_dir}
is_resumed=false #true
use_pretrained=1 # loading only the AE but not the LDM as pretrained models (from source_model)
source_model="stabilityai/stable-diffusion-2-1-base"
torch_detect_anomaly=0 
img_width=640 #224
img_height=640 #224
#clip_grad_norm_by=15.0
#clip_grad_norm_or_value='value'
controlnet_conditioning_scale=1.0

python3 src/python/training/train_controlnet.py \
        --seed ${seed} \
        --run_dir ${run_dir} \
        --training_ids ${training_ids} \
        --validation_ids ${validation_ids} \
        --stage1_uri=${stage1_uri} \
        --ddpm_uri ${ddpm_uri} \
        --config_file ${config_file} \
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
        --scale_factor=${scale_factor} \
        --controlnet_conditioning_scale=${controlnet_conditioning_scale} \
        --"is_ldm_fine_tuned" \
        --"init_from_unet"
        #--"is_resumed" \
        #--"use_default_report_text" \
        #--"is_stage1_fine_tuned" \
