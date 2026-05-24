"""
chargement et traitement du dataset Flickr8k pour les deux phases du projet

"""

import os
import random
from collections import defaultdict

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config



def get_transform(split: str):
    """
    transformer les images selon les splits
    - train : data augmentation légère (flip, color jitter)
    - val/test : resize + center crop uniquement
    """
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],   # stats CLIP
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    if split == "train":
        return T.Compose([
            T.Resize(config.IMAGE_SIZE + 32),
            T.RandomCrop(config.IMAGE_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.ToTensor(),
            normalize,
        ])
    else:
        return T.Compose([
            T.Resize(config.IMAGE_SIZE + 32),
            T.CenterCrop(config.IMAGE_SIZE),
            T.ToTensor(),
            normalize,
        ])


def load_captions(captions_file: str) -> dict[str, list[str]]:
    """
    Lit captions.txt et retourne un dict { image_name: [cap1, cap2, ...] }.
    Compatible avec le format Flickr8k (image.jpg#0\tcaption) et format Kaggle (CSV).
    """
    try:
        df = pd.read_csv(captions_file, sep="\t", header=None)
        if df.shape[1] != 2:
            raise ValueError("Format TAB séparé attendait 2 colonnes")
        df.columns = ["image", "caption"]
    except:
        try:
            df = pd.read_csv(captions_file)
            df.columns = [c.strip().lower() for c in df.columns]
        except:
            raise ValueError(f"Impossible de lire {captions_file}")

    captions = defaultdict(list)
    for _, row in df.iterrows():
        img_name = str(row["image"]).strip()
        caption  = str(row["caption"]).strip()
        # Certaines versions ont "image.jpg#0" → on garde juste "image.jpg"
        img_name = img_name.split("#")[0]
        captions[img_name].append(caption)

    return dict(captions)

def make_splits(
    captions: dict[str, list[str]],
    train_size: int = config.TRAIN_SIZE,
    val_size:   int = config.VAL_SIZE,
    test_size:  int = config.TEST_SIZE,
    seed:       int = config.RANDOM_SEED,
    images_dir: str = config.IMAGES_DIR,
) -> tuple[list[str], list[str], list[str]]:
    """
    Découpe la liste des images uniques en train / val / test.
    Filtre les images qui n'existent pas physiquement.
    Retourne trois listes de noms de fichiers image.
    """

    existing_images = []
    for img_name in sorted(captions.keys()):
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            existing_images.append(img_name)
    
    random.seed(seed)
    random.shuffle(existing_images)

    n = len(existing_images)
    print(f"✓ {n} images trouvées (filtrées par existence physique)")
    assert train_size + val_size + test_size <= n, (
        f"Le dataset contient seulement {n} images uniques ; "
        f"splits demandés = {train_size}+{val_size}+{test_size}={train_size+val_size+test_size}"
    )

    train_imgs = existing_images[:train_size]
    val_imgs   = existing_images[train_size : train_size + val_size]
    test_imgs  = existing_images[train_size + val_size : train_size + val_size + test_size]
    return train_imgs, val_imgs, test_imgs


# pour chaque phase on a un dataset dédié, avec des formats de sortie différents adaptés à la tâche

#phase 1 : encodage contrastif (CLIP-like) => (image_tensor, caption_str)
class Flickr8kContrastiveDataset(Dataset):
    """
    Retourne des paires (image_tensor, caption_str).

    Pour chaque image on choisit UNE caption aléatoire parmi les 5 disponibles
    (ou la caption à l'index `caption_idx` si fourni).  Pendant l'entraînement
    la diversité entre epochs est assurée par le tirage aléatoire.
    """

    def __init__(
        self,
        image_names:  list[str],
        captions:     dict[str, list[str]],
        images_dir:   str  = config.IMAGES_DIR,
        split:        str  = "train",
        caption_idx:  int | None = None,
    ):
        self.image_names = image_names
        self.captions    = captions
        self.images_dir  = images_dir
        self.split       = split
        self.caption_idx = caption_idx
        self.transform   = get_transform(split)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caps = self.captions[img_name]
        if self.caption_idx is not None:
            caption = caps[self.caption_idx % len(caps)]
        else:
            caption = random.choice(caps)

        return image, caption


#phase 2 : fine-tuning de GPT-2 => (image_tensor, input_ids, attention_mask, labels)

class Flickr8kCaptionDataset(Dataset):
    """
    Retourne des tuples (image_tensor, input_ids, attention_mask, labels)
    pour le fine-tuning de GPT-2.

    Chaque image × caption = 1 exemple → 5 exemples par image.
    Format du texte injecté dans GPT-2 :
        <|startoftext|> <caption> <|endoftext|>
    Les labels sont les input_ids décalés d'un token (teacher forcing).
    Les positions du prefix image sont masquées dans les labels (=-100)
    pour ne pas calculer de loss sur le token de début.
    """

    def __init__(
        self,
        image_names:  list[str],
        captions:     dict[str, list[str]],
        tokenizer,                           # GPT2Tokenizer
        images_dir:   str  = config.IMAGES_DIR,
        split:        str  = "train",
        max_len:      int  = config.MAX_CAPTION_LEN,
    ):
        self.images_dir = images_dir
        self.tokenizer  = tokenizer
        self.transform  = get_transform(split)
        self.max_len    = max_len

        # Expansion : une ligne par (image, caption)
        self.samples: list[tuple[str, str]] = []
        for name in image_names:
            for cap in captions[name]:
                self.samples.append((name, cap))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch

        img_name, caption = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Tokenisation
        text = config.BOS_TOKEN + " " + caption + " " + config.EOS_TOKEN
        enc  = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)      # (max_len,)
        attention_mask = enc["attention_mask"].squeeze(0)  # (max_len,)

        # Labels : on copie les input_ids et on masque les tokens de padding
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100   # ignoré dans le cross-entropy

        return image, input_ids, attention_mask, labels


def build_datasets(
    phase: str = "phase1",   # "phase1" ou "phase2"
    tokenizer=None,          # requis si phase == "phase2"
) -> tuple:
    """
    Construit les trois datasets (train/val/test) pour la phase demandée.
    Retourne (train_ds, val_ds, test_ds).
    """
    captions = load_captions(config.CAPTIONS_FILE)
    train_imgs, val_imgs, test_imgs = make_splits(captions)

    if phase == "phase1":
        train_ds = Flickr8kContrastiveDataset(train_imgs, captions, split="train")
        val_ds   = Flickr8kContrastiveDataset(val_imgs,   captions, split="val")
        test_ds  = Flickr8kContrastiveDataset(test_imgs,  captions, split="test")
    elif phase == "phase2":
        assert tokenizer is not None, "tokenizer requis pour la phase 2"
        train_ds = Flickr8kCaptionDataset(train_imgs, captions, tokenizer, split="train")
        val_ds   = Flickr8kCaptionDataset(val_imgs,   captions, tokenizer, split="val")
        test_ds  = Flickr8kCaptionDataset(test_imgs,  captions, tokenizer, split="test")
    else:
        raise ValueError(f"phase doit être 'phase1' ou 'phase2', reçu: {phase!r}")

    return train_ds, val_ds, test_ds


def collate_phase1(batch):
    """Collate pour Flickr8kContrastiveDataset : (images, captions_list)."""
    import torch
    images   = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]
    return images, captions


def collate_phase2(batch):
    """Collate pour Flickr8kCaptionDataset : (images, input_ids, attn, labels)."""
    import torch
    images   = torch.stack([b[0] for b in batch])
    ids      = torch.stack([b[1] for b in batch])
    masks    = torch.stack([b[2] for b in batch])
    labels   = torch.stack([b[3] for b in batch])
    return images, ids, masks, labels
