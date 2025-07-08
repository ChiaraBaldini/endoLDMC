### Inerence: generate a new endoscopic image based on noise, caption and lesion BB mask ###
# This script is used to run the inference of the pre-trained LDM + ControlNet model

seed=42
num_inference_steps=200 #1000 #200
y_size=80 #80 for 640, 28 for 224  (64 #28 # 64 #28 #64 #20 # size of noise embedding, e.g. noisy_e shape: torch.Size([3, 4, 64, 64]))
x_size=80 #80 for 640, 28 for 224
experiment="INF_LDM_controlnet_txt_bb_ft"
guidance_scale=1.0 #0.01 #1.4 # Test different text-guidance scales to give more or less importance to the text prompt
scheduler_type=ddpm #ddpm #ddpm #ddpm #ddim 
controlnet_conditioning_scale=1.0 #0.8 #1.4 #1.0 # Test different controlnet scales to give more or less importance to the lesion BB mask
output_dir="project/outputs/inference/${experiment}/actually_ddpm/gs1_cgs1_200steps/"
ddpm_uri="project/outputs/runs/LDM_txt_v0/diffusion_best_model.pth"
controlnet_uri="project/outputs/runs/CTRL_CCNET_ANY_ACQ_TXT_v0/controlnet_best_model.pth"
stage1_uri=""
test_ids="project/outputs/ids/test.tsv"
config_file="configs/controlnet/controlnet_v1.yaml"
scale_factor=0.1 #0.1 #0.01 #=0.3
batch_size=1 #8 #16 
num_workers=8 #16 #64
use_pretrained=1 # loading only the AE but not the LDM as pretrained models (from source_model)
source_model="stabilityai/stable-diffusion-2-1-base"
img_width=640 #224
img_height=640 #224

python3 src/python/training/run_controlnet_inference.py \
      --seed ${seed} \
      --output_dir ${output_dir} \
      --test_ids ${test_ids} \
      --stage1_uri=${stage1_uri} \
      --ddpm_uri ${ddpm_uri} \
      --controlnet_uri ${controlnet_uri} \
      --config_file ${config_file} \
      --batch_size ${batch_size} \
      --num_workers ${num_workers} \
      --experiment ${experiment} \
      --use_pretrained ${use_pretrained} \
      --source_model ${source_model} \
      --img_width ${img_width} \
      --img_height ${img_height} \
      --scale_factor=${scale_factor} \
      --controlnet_conditioning_scale=${controlnet_conditioning_scale} \
      --num_inference_steps ${num_inference_steps} \
      --y_size ${y_size} \
      --x_size ${x_size} \
      --scheduler_type ${scheduler_type} \
      --guidance_scale ${guidance_scale} \
      #--"init_from_unet" \
      #--"is_stage1_fine_tuned"
      #--"is_ldm_fine_tuned"
      #--"use_default_report_text" \


