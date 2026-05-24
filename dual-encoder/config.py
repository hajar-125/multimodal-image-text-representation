"""
config.py – hyperparamètres et chemins centralisés pour la Solution 3.
Dual Encoder (Phase 1) + GPT-2 Decoder (Phase 2) sur Flickr8k.
"""

import os

# ── Chemins ───────────────────────────────────────────────────────────────────
DATA_ROOT       = "flickr8K"          # dossier contenant Flicker8k_Dataset/ et Flickr8k.token.txt
IMAGES_DIR      = os.path.join(DATA_ROOT, "Flicker8k_Dataset")
CAPTIONS_FILE   = os.path.join(DATA_ROOT, "Flickr8k.token.txt")

CHECKPOINT_DIR  = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PHASE1_CKPT     = os.path.join(CHECKPOINT_DIR, "dual_encoder.pt")
PHASE2_CKPT     = os.path.join(CHECKPOINT_DIR, "decoder.pt")

# ── Splits standard Flickr8k ──────────────────────────────────────────────────
TRAIN_SIZE      = 6000
VAL_SIZE        = 1000
TEST_SIZE       = 1000
RANDOM_SEED     = 42

# ── Image ─────────────────────────────────────────────────────────────────────
IMAGE_SIZE      = 224          # taille d'entrée du ViT

# ── Encodeurs ─────────────────────────────────────────────────────────────────
# open_clip: modèle ViT-B/32 pré-entraîné sur LAION
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"     # poids CLIP originaux OpenAI
EMBED_DIM       = 512          # dimension de l'espace partagé

# Encodeur texte BERT (pour phase 1 seulement)
BERT_MODEL_NAME = "bert-base-uncased"
MAX_TEXT_LEN    = 77           # même longueur que CLIP

# ── Phase 1 – entraînement contrastif ─────────────────────────────────────────
PHASE1_EPOCHS        = 20
PHASE1_BATCH_SIZE    = 64
PHASE1_LR            = 1e-4
PHASE1_WEIGHT_DECAY  = 1e-4
PHASE1_TEMPERATURE   = 0.07    # température InfoNCE

# ── Phase 2 – fine-tuning GPT-2 ───────────────────────────────────────────────
GPT2_MODEL_NAME      = "gpt2"  # gpt2-small (~117M paramètres)
MAX_CAPTION_LEN      = 64      # longueur max des captions générées

PHASE2_EPOCHS        = 15
PHASE2_BATCH_SIZE    = 32
PHASE2_LR            = 5e-5
PHASE2_WEIGHT_DECAY  = 1e-4

# Token spécial de début de caption injecté avant le texte
BOS_TOKEN           = "<|startoftext|>"
EOS_TOKEN           = "<|endoftext|>"

# ── Génération (inférence) ────────────────────────────────────────────────────
BEAM_SIZE            = 5
MAX_NEW_TOKENS       = 50

# ── Device ────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
