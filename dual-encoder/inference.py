"""
inference.py – Génération de caption pour une image quelconque.

Usage :
    python inference.py --image path/to/image.jpg
    python inference.py --image path/to/image.jpg --strategy beam --beams 5
    python inference.py --image path/to/image.jpg --strategy sample --temperature 0.8
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torchvision.transforms as T
from PIL import Image

import config
from decoder import CaptioningModel, build_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Transform (identique à val/test)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image_path: str) -> torch.Tensor:
    transform = T.Compose([
        T.Resize(config.IMAGE_SIZE + 32),
        T.CenterCrop(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std= [0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)   # (1, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Génération de caption pour une image")
    parser.add_argument("--image",       required=True,             help="Chemin de l'image")
    parser.add_argument("--p1_ckpt",     default=config.PHASE1_CKPT)
    parser.add_argument("--p2_ckpt",     default=config.PHASE2_CKPT)
    parser.add_argument("--strategy",    default="beam",            choices=["beam", "greedy", "sample"])
    parser.add_argument("--beams",       type=int,   default=config.BEAM_SIZE)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens",  type=int,   default=config.MAX_NEW_TOKENS)
    args = parser.parse_args()

    device = torch.device(config.DEVICE)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = build_tokenizer()

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = CaptioningModel(
        phase1_ckpt=args.p1_ckpt,
        prefix_len=10,
        mapper_variant="mlp",
        device=config.DEVICE,
    ).to(device)
    model.gpt2.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(args.p2_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Image ─────────────────────────────────────────────────────────────────
    image = preprocess(args.image).to(device)

    # ── Génération ────────────────────────────────────────────────────────────
    do_sample = args.strategy == "sample"
    num_beams = args.beams if args.strategy == "beam" else 1

    caption = model.generate_caption(
        image,
        tokenizer,
        max_new_tokens=args.max_tokens,
        num_beams=num_beams,
        temperature=args.temperature,
        do_sample=do_sample,
    )

    print(f"\nImage   : {args.image}")
    print(f"Caption : {caption}\n")
    return caption


if __name__ == "__main__":
    main()
