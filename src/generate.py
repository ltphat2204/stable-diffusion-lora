import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Generate image with fine-tuned LoRA adapter")
    parser.add_argument(
        "--model_dir", type=str,
        default="lora-dong-ho-style",
        help="Local folder chứa adapter_config.json và weights"
    )
    parser.add_argument(
        "--pretrained", type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model"
    )
    parser.add_argument(
        "--prompt", type=str,
        default="a dreamy fantasy landscape at sunrise",
        help="Generation prompt"
    )
    parser.add_argument(
        "--output", type=str,
        default="generated.png",
        help="File ảnh đầu ra"
    )
    parser.add_argument(
        "--steps", type=int,
        default=50,
        help="Số inference steps"
    )
    parser.add_argument(
        "--scale", type=float,
        default=7.5,
        help="Guidance scale"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained,
        torch_dtype=torch.float16
    ).to(device)

    model_dir = os.path.abspath(args.model_dir)
    pipe.unet = PeftModel.from_pretrained(
        pipe.unet,
        model_dir,
        local_files_only=True
    ).to(device)

    image = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.scale
    ).images[0]

    image.save(args.output)
    print(f"Saved → {args.output}")

if __name__ == "__main__":
    main()
