# Conditional DDPM for iCLEVR Image Generation

This repository implements a Conditional Denoising Diffusion Probabilistic Model (DDPM) to generate synthetic images based on multi-hot object labels. It features a custom U-Net with AdaGN (Adaptive Group Normalization) conditioning, DDIM sampling, and Classifier-Free Guidance (CFG).

## Directory Structure

```text
.
├── configs/
│   └── config.yaml        # Centralized hyperparameters and paths
├── data/                  # Dataset directory
├── diffusion/             # Noise scheduling and DDIM/DDPM samplers
├── evaluation/            # Evaluation scripts
├── models/                # U-Net, ResBlocks, Attention, and embeddings
├── utils/                 # EMA, early stopping, and checkpointing helpers
├── train.py               # Main training loop
├── inference.py           # Evaluation and image generation script
└── requirements.txt       # Python dependencies
```

## Setup Instructions

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset and Evaluator Checkpoint
The model expects the iCLEVR dataset and associated JSON label files to be located in the `data/` directory as defined in `configs/config.yaml`.

1. Run the dataset fetch script:
```bash
bash get_dataset.sh
```

2. Download the checkpoint:
```bash
bash get_checkpoint.sh
```

## Training

To train the model from scratch using the default configuration:
```bash
python3 train.py --config configs/config.yaml
```

If training was interrupted, or you wish to fine-tune, pass the ```--resume` flag:

```bash
python3 train.py --config configs/config.yaml --resume ckpts/ckpt_epoch050.pt
```

Checkpoints will be saved to the `./ckpts/` directory. 

## Inference

```bash
# Example command
python3 inference.py --config configs/config.yaml --ckpt ckpts/best.pt
```

