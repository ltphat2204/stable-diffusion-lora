import os
import time
import random
import argparse
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class DongHoDataset(Dataset):
    def __init__(self, folder, resolution):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained",    type=str,   default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--dataset_dir",   type=str,   default="data/processed")
    p.add_argument("--output_dir",    type=str,   default="lora-dong-ho-style")
    p.add_argument("--resolution",    type=int,   default=512)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--epochs",        type=int,   default=5)
    p.add_argument("--lora_r",        type=int,   default=4)
    p.add_argument("--lora_alpha",    type=int,   default=16)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained, subfolder="text_encoder") \
                     .to(device, dtype=torch.float16).eval()
    with torch.no_grad():
        tok = tokenizer([""], padding="max_length",
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt")
        emb = text_encoder(tok.input_ids.to(device))[0]

    vae = AutoencoderKL.from_pretrained(
        args.pretrained, subfolder="vae", torch_dtype=torch.float16
    ).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained, subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    scheduler = DDPMScheduler.from_pretrained(args.pretrained, subfolder="scheduler")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian"
    )
    unet = get_peft_model(unet, lora_cfg).to(device).train()
    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
    except:
        pass

    dataset = DongHoDataset(args.dataset_dir, args.resolution)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    accelerator = Accelerator(mixed_precision="fp16")
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images in pbar:
            images = images.to(device)
            optimizer.zero_grad()
            with accelerator.autocast():
                latents   = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
                noise     = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                          (latents.size(0),), device=device)
                noisy_lat = scheduler.add_noise(latents, noise, timesteps)
                pred      = unet(noisy_lat, timesteps,
                                 encoder_hidden_states=emb.repeat(latents.size(0),1,1)).sample
                loss      = F.mse_loss(pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch} done in {time.time() - start:.1f}s")

    unet.save_pretrained(args.output_dir)
    print(f"Training finished, model saved at {args.output_dir}")

if __name__ == "__main__":
    main()
