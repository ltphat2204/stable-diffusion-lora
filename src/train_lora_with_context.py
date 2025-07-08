import os
import time
import random
import argparse
import logging
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm import tqdm

from transformers import (
    CLIPTextModel, CLIPTokenizer,
    CLIPVisionModel, CLIPImageProcessor
)
from diffusers import (
    AutoencoderKL, UNet2DConditionModel,
    DDPMScheduler, StableDiffusionPipeline
)
from peft import LoraConfig, get_peft_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class DongHoDataset(Dataset):
    def __init__(self, img_folder, txt_folder, resolution):
        self.samples = []
        for fn in os.listdir(img_folder):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                base = os.path.splitext(fn)[0]
                img_p = os.path.join(img_folder, fn)
                txt_p = os.path.join(txt_folder, base + ".txt")
                if os.path.isfile(txt_p):
                    cap = open(txt_p, encoding="utf-8").read().strip()
                    self.samples.append((img_p, cap))
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, cap = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        return self.transform(img), cap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained",   type=str,   default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--img_dir",      type=str,   default="data/processed")
    parser.add_argument("--txt_dir",      type=str,   default="data/raw")
    parser.add_argument("--output_dir",   type=str,   default="lora-dong-ho-style")
    parser.add_argument("--resolution",   type=int,   default=512)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--lora_r",       type=int,   default=4)
    parser.add_argument("--lora_alpha",   type=int,   default=16)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--eval_samples", type=int,   default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # CLIP text + vision
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained, subfolder="text_encoder") \
                       .to(device).eval()
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32") \
                       .to(device).eval()
    img_proc     = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Diffusion models (VAE half, UNet half)
    vae       = AutoencoderKL.from_pretrained(args.pretrained, subfolder="vae",
                  torch_dtype=torch.float16).to(device).eval()
    unet      = UNet2DConditionModel.from_pretrained(args.pretrained, subfolder="unet",
                  torch_dtype=torch.float16).to(device)
    scheduler = DDPMScheduler.from_pretrained(args.pretrained, subfolder="scheduler")

    # Inject LoRA into UNet
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["to_q","to_k","to_v","to_out.0"],
        init_lora_weights="gaussian"
    )
    unet = get_peft_model(unet, lora_cfg).to(device).train()
    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
    except:
        pass

    # Dataset + split
    dataset = DongHoDataset(args.img_dir, args.txt_dir, args.resolution)
    split = int(0.9 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset)-split])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Optimizer + Accelerator
    optimizer   = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    accelerator = Accelerator(mixed_precision="fp16")
    unet, optimizer, train_loader = accelerator.prepare(unet, optimizer, train_loader)

    # Prepare pipeline for eval, cast VAE decode to float32 to avoid dtype mismatch
    pipe = StableDiffusionPipeline.from_pretrained(
               args.pretrained, torch_dtype=torch.float16
           ).to(device)
    pipe.unet = unet
    pipe.vae = pipe.vae.to(device, dtype=torch.float32)

    # Training loop
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        for images, caps in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            optimizer.zero_grad()
            with accelerator.autocast():
                latents   = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
                noise     = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                          (latents.size(0),), device=device)
                noisy_lat = scheduler.add_noise(latents, noise, timesteps)
                tok       = tokenizer(caps, padding="max_length", truncation=True,
                                      max_length=tokenizer.model_max_length,
                                      return_tensors="pt").to(device)
                txt_emb   = text_encoder(tok.input_ids)[0]
                pred      = unet(noisy_lat, timesteps, encoder_hidden_states=txt_emb).sample
                loss      = F.mse_loss(pred, noise)
            accelerator.backward(loss)
            optimizer.step()

        logger.info(f"Epoch {epoch} done in {time.time() - t0:.1f}s")

        # CLIP-based eval
        sims = []
        n_eval = min(args.eval_samples, len(val_ds))
        for i in range(n_eval):
            _, cap = val_ds[i]
            gen = pipe(cap, num_inference_steps=20, guidance_scale=7.5).images[0]
            proc = img_proc(images=gen, return_tensors="pt").to(device)
            vis = vision_model(**proc).pooler_output
            tok2 = tokenizer([cap], padding=True, truncation=True,
                              max_length=tokenizer.model_max_length,
                              return_tensors="pt").to(device)
            txt = text_encoder(tok2.input_ids)[0][:,0,:]
            vis_n = vis / vis.norm(dim=-1, keepdim=True)
            txt_n = txt / txt.norm(dim=-1, keepdim=True)
            sims.append((vis_n * txt_n).sum().item())
        score = sum(sims) / len(sims)
        logger.info(f"Epoch {epoch} CLIP score: {score:.4f}")

    unet.save_pretrained(args.output_dir)
    logger.info(f"Training finished, LoRA saved at {args.output_dir}")

if __name__ == "__main__":
    main()
