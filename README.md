# Joint Visual-Textual Representation Learning for Image Captioning
## A Contrastive and Generative Approach on Flickr8K

### Architecture
- **Phase 1** : Dual Encoder contrastif — ViT-B/32 (open_clip) + CLIP text encoder + InfoNCE loss
- **Phase 2** : Décodeur génératif — GPT-2 small avec prefix visual token

### Dataset
Flickr8K — 8 000 images, 5 captions chacune, split 6k/1k/1k

### Référence théorique
Inspiré de BLIP (Li et al., 2022)

### Structure
dual-encoder/
├── config.py
├── dataset.py
├── dual_encoder.py
├── train_phase1.py
├── train_phase2.py
├── decoder.py
├── evaluate.py
└── inference.py

### Installation
pip install -r requirements.txt

### Lancer l'entraînement Phase 1
python dual-encoder/train_phase1.py
