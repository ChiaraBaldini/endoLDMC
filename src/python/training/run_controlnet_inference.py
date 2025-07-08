""" Script to generate sample images from the diffusion model with BB+text conditioning. In the generation of the images, the script is using a DDIM scheduler."""
from __future__ import annotations
import argparse
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from monai.utils import set_determinism
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from monai.config import print_config
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from util import get_test_dataloader, setup_logger, get_models_for_controlnet, get_scale_factor, sample_from_ldm, sample_from_controlnet, save_controlnet_inference_sample, load_controlnet
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Path to save the generated images")
    parser.add_argument("--experiment", help="Experiment name to keep track.")
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--upper_limit", default=None, help="Maximum number of images to be generated.")

    parser.add_argument("--source_model", type=str, default="stabilityai/stable-diffusion-2-1-base", help="source model for the stage1 autoencoder and text_encoder")
    parser.add_argument("--stage1_uri", help="Path readable by load_model.")
    parser.add_argument("--ddpm_uri", help="Path readable by load_model.")
    parser.add_argument("--controlnet_uri", default="controlnet_best_model.pth", help="Path readable by load_model.")
    parser.add_argument("--init_from_unet", action="store_true" , help="If true, the controlnet will be initialized from the unet of the diffusion model. Otherwise, the controlnet will be initialized from monai generative models.")
    parser.add_argument("--use_pretrained", type=int, default=0, help="use a pretrained stage1 autoencoder instead of the trained checkpoint. 0 = False, 1 = stage 1 only, 2=stage 1 and diffusion model")
    parser.add_argument("--is_stage1_fine_tuned", action="store_true" , help="Info if the stage1 model was fine tuned, therefore requiring different ldm input and output dims.")
    parser.add_argument("--is_ldm_fine_tuned", action="store_true" , help="Info if the ldm model was fine tuned, therefore requiring different controlnet input and output dims.")

    parser.add_argument("--cond_on_acq_times", action="store_true" , help="If true, MRI acquisition times will be passed as conditional into controlnet during train and eval.")
    parser.add_argument("--use_default_report_text", action="store_true" , help="If true, the default report text will be used for all samples returned from the dataloader. Otherwise, a custom report text will be loaded from the from the dataloader.")
    parser.add_argument("--num_inference_steps", type=int, help="")
    parser.add_argument("--x_size", type=int, default=64, help="Latent space x size.")
    parser.add_argument("--y_size", type=int, default=64, help="Latent space y size.")
    parser.add_argument("--guidance_scale", type=float, default=None, help="previous default=7.0. Multiplier for textual conditioning input. The higher, the more adherence to text.")
    parser.add_argument("--controlnet_conditioning_scale", type=float,  default=1.0, help="The scaling factor for the conditioning output to the controlnet. This is used to scale the output of the controlnet. Default is 1.0. It is the multiplier of controlnet latents before adding them to ldm latents")
    parser.add_argument("--scale_factor", type=float, default=0.01, help="Latent space y size.")
    parser.add_argument("--scheduler_type", type=str, default=None, help="Which scheduler to use for inference. e.g.: 'ddim' or 'ddpm'")

    parser.add_argument("--test_ids", help="Location of file with test ids.")
    parser.add_argument("--img_height", type=int,  default=512, help="The image height that the dataloader will resize the input images to")
    parser.add_argument("--img_width", type=int,  default=512, help="The image width that the dataloader will resize the input images to")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, help="")

    args = parser.parse_args()
    return args


def main(args):
    FILENAME = 'filename'
    CAPTION = 'caption'
    CAPTION_RAW= 'caption'
    SOURCE = 'image'
    MASK = 'mask'

    set_determinism(seed=args.seed)
    print_config()
    output_dir = Path(args.output_dir)
    if not output_dir.exists(): output_dir.mkdir(exist_ok=True, parents=True)

    setup_logger(output_dir / f"run_inference_controlnet_{args.experiment}.log")

    logging.info(f"Output directory: {str(output_dir)}")
    logging.info(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        logging.info(f"  {k}: {v}")

    device = torch.device("cuda") # Use GPU for inference

    # Loading the configuration file that contains ldm, controlnet and stage1 parameters
    config = OmegaConf.load(args.config_file)

    # Initializing the scheduler for the diffusion process
    if args.scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
            beta_start=config["ldm"]["scheduler"]["beta_start"],
            beta_end=config["ldm"]["scheduler"]["beta_end"],
            schedule=config["ldm"]["scheduler"]["schedule"],
            prediction_type=config["ldm"]["scheduler"]["prediction_type"],
            clip_sample=False,
        )
        scheduler.set_timesteps(num_inference_steps = args.num_inference_steps)
        logging.info(f"Using {scheduler.__class__.__name__} scheduler with {args.num_inference_steps} num_inference_steps.")
    elif args.scheduler_type == "ddpm":
        scheduler = DDPMScheduler(**config["ldm"].get("scheduler", dict()))
    else:
        raise NotImplementedError(f"scheduler {args.scheduler_type} is not implemented. Please use either 'ddpm' or 'ddim'.")

    # Initializing the LDM and STAGE1 models using the configuration and scheduler
    config, stage1, diffusion, _ = get_models_for_controlnet(config=config, args=args, scheduler=scheduler, device=device)

    # Initializing the ControlNet module
    logging.info(f"Loading controlnet model from config {config}.")
    controlnet = load_controlnet(config, args, diffusion, device)

    logging.info(f"Loading text_encoder model: CLIPTestModel from {args.source_model}.")

    # Initializing the text encoder and tokenizer (for text conditioning)
    text_encoder = CLIPTextModel.from_pretrained(args.source_model, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.source_model, subfolder="tokenizer")

    # Preparation of inference
    stage1 = stage1.to(device)
    diffusion = diffusion.to(device)
    controlnet = controlnet.to(device)
    text_encoder = text_encoder.to(device)

    stage1.eval()
    diffusion.eval()
    controlnet.eval()
    text_encoder.eval()

    empty_prompt = [""]
    text_inputs = tokenizer(
        empty_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    empty_prompt_embeds = text_encoder(text_input_ids.squeeze(1).to(device))
    empty_prompt_embeds = empty_prompt_embeds[0].to(device)
    logging.debug(f"empty_prompt_embeds: {empty_prompt_embeds}")

    test_loader = get_test_dataloader(
        batch_size=args.batch_size,
        test_ids=args.test_ids,
        num_workers=args.num_workers,
        # upper_limit=None if args.upper_limit is None else int(args.upper_limit),
        img_height = args.img_height,
        img_width = args.img_width,
        use_default_report_text= args.use_default_report_text
    )

    args.scale_factor = get_scale_factor(stage1=stage1) if args.scale_factor is None else args.scale_factor

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, x in pbar:
        if i == 937 or i == 938 or i == 939:
            logging.info(f"{i} Input: {x[FILENAME]}, Output: {x[FILENAME]}")
        final_file_name = x[FILENAME][0]
        print(final_file_name)

        final_file_name = Path(final_file_name).name.replace('.jpg', '_synthetic.png')
        if Path(output_dir / "PIL" / final_file_name).exists() or Path(output_dir / final_file_name).exists():
            logging.debug(f"Image {final_file_name} already exists in the output dir {output_dir}. Skipping.")
            continue
        else:
            logging.debug(f"Image {final_file_name} does not exist in the output dir {output_dir}. Generating...")

        images = x[SOURCE].to(device)
        reports = x[CAPTION].to(device)
        reports_raw = x[CAPTION_RAW] # not a tensor
        cond = x[MASK].to(device)

        prompt_embeds = text_encoder(reports.squeeze(1))
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = torch.cat((empty_prompt_embeds, prompt_embeds), dim=0)
        logging.debug(f"concat prompt_embeds: {prompt_embeds}")

        # Verify shape of a mask loaded from path
        print(cond.shape)
        if cond.shape[1]==3:
            cond=cond[:,0,:,:].unsqueeze(1)
            print(cond.shape)

        # Batch size of 1 for noise
        noise = torch.randn((1, config["controlnet"]["params"]["in_channels"], args.x_size, args.y_size)).to(device)

        with autocast(enabled=True):
            with torch.no_grad():
                progress_bar = tqdm(scheduler.timesteps)
                for t in progress_bar:
                    timesteps = torch.Tensor((t,)).to(noise.device).long()
                    noise_input = torch.cat([noise] * 2)
                    cond_input = torch.cat([cond] * 2)
                    logging.debug(f"noise_input shape for sampling from controlnet at timestep {t}: {noise_input.shape}.")

                    down_block_res_samples, mid_block_res_sample = sample_from_controlnet(
                        controlnet=controlnet,
                        noisy_e=noise_input,
                        controlnet_cond=cond_input,
                        timesteps=timesteps,
                        prompt_embeds=prompt_embeds,
                        conditioning_scale=args.controlnet_conditioning_scale,
                    )

                    model_output = sample_from_ldm(
                        model=diffusion,
                        noisy_e=noise_input,
                        timesteps=timesteps,
                        prompt_embeds=prompt_embeds,
                        down_block_res_samples=down_block_res_samples,
                        mid_block_res_sample=mid_block_res_sample,
                    )
                    noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                    if args.guidance_scale is not None:
                        if args.guidance_scale == 0:
                            noise_pred = noise_pred_uncond
                        else:
                            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_text
                    noise, _ = scheduler.step(noise_pred, t, noise)

        with torch.no_grad():
            sample = stage1.model.decode(noise / args.scale_factor)

        save_controlnet_inference_sample(run_dir=output_dir, filename=final_file_name, sample=sample, cond=cond, images=images, save_matplotlib=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)

