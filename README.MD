# Stable Diffusion LoRA Fine-Tuning Project

A minimal setup for fine-tuning Stable Diffusion using LoRA adapters to learn a custom "dreamy landscape" style.

## Prerequisites

* Python 3.10 or later
* Git
* NVIDIA GPU with CUDA & cuDNN configured

## Installation

1. Clone this repository:

   ```bash
   git clone <your-repo-url> stable-diffusion-lora
   cd stable-diffusion-lora
   ```
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   # venv\Scripts\activate  # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
stable-diffusion-lora/
├── venv/                   # Python virtual environment
├── data/
│   ├── raw/                # Original images (downloaded)
│   └── processed/          # Preprocessed images (512×512, normalized)
├── scripts/
│   └── download_data.py    # (Optional) script to fetch images from Unsplash
├── src/
│   ├── dataset.py          # Defines PyTorch Dataset & DataLoader
│   └── train_lora.py       # Script to fine-tune LoRA adapters
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md               # This file
```

## Data Preparation

1. Place your gathered "dreamy landscape" images into `data/raw/`.
2. Run the preprocessing script (to be provided in next steps) to resize and normalize images into `data/processed/`.
