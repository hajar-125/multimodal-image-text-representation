# Solution 3 – Dual Encoder + GPT-2 Decoder (Flickr8k)

Implémentation de la Solution 3 : encodeur dual contrastif (Phase 1) + décodeur GPT-2 conditionné par image (Phase 2), inspiré de BLIP/ClipCap.

## Architecture

```
Phase 1 — Dual Encoder contrastif
    Image : ViT-B/32 (open_clip, pré-entraîné CLIP) → ProjectionHead → embed (512)
    Texte : CLIP text encoder                        → ProjectionHead → embed (512)
    Loss  : InfoNCE symétrique (image↔texte)

Phase 2 — Caption Decoder
    Image encoder (gelé depuis Phase 1)
    → ImagePrefixMapper (MLP) → 10 tokens visuels (768 dim)
    → [prefix | <|startoftext|> caption <|endoftext|>]
    → GPT-2 small fine-tuné avec causal LM loss
```

## Dataset

Flickr8k (Kaggle) — format attendu :
```
data/flickr8k/
    Images/          ← 8091 fichiers .jpg
    captions.txt     ← colonnes: image,caption
```

Splits : 6000 train / 1000 val / 1000 test (sur images uniques).

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Phase 1 – Entraînement dual encoder

```bash
python train/train_phase1.py
```
Checkpoint sauvegardé dans `checkpoints/dual_encoder.pt`.

### Phase 2 – Fine-tuning GPT-2

```bash
python train/train_phase2.py
```
Checkpoint sauvegardé dans `checkpoints/decoder.pt`.

### Évaluation (BLEU / METEOR / ROUGE-L / CIDEr)

```bash
python evaluate.py
python evaluate.py --n 200          # évaluation rapide sur 200 images
python evaluate.py --output results/scores.json
```

### Inférence sur une image quelconque

```bash
# Beam search (défaut)
python inference.py --image path/to/image.jpg

# Greedy decoding
python inference.py --image path/to/image.jpg --strategy greedy

# Sampling créatif
python inference.py --image path/to/image.jpg --strategy sample --temperature 0.8
```

## Résultats attendus sur Flickr8k (test set)

| Métrique | Score typique |
|----------|--------------|
| BLEU-1   | ~0.62        |
| BLEU-4   | ~0.22        |
| METEOR   | ~0.23        |
| ROUGE-L  | ~0.45        |
| CIDEr    | ~0.75        |

## Références

- BLIP: Bootstrapping Language-Image Pre-training — Li et al. 2022 (arxiv:2201.12086)
- ClipCap: CLIP Prefix for Image Captioning — Mokady et al. 2021
- open_clip: https://github.com/mlfoundations/open_clip
