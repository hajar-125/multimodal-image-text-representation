"""
train/train_phase1.py – Entraînement Phase 1 : Dual Encoder contrastif.

Pipeline :
    1. Charge Flickr8k (split train / val)
    2. Initialise DualEncoder (ViT-B/32 CLIP gelé + projection heads)
    3. Entraîne avec InfoNCE symétrique
    4. Sauvegarde le meilleur checkpoint selon la val loss

Lancement :
    python train/train_phase1.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from dataset import build_datasets, collate_phase1
from dual_encoder import DualEncoder, contrastive_loss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, training=True):
    model.train(training)
    total_loss = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, captions in tqdm(loader, leave=False, desc="train" if training else "val"):
            images = images.to(device)

            img_emb, txt_emb, logit_scale = model(images, captions)
            loss = contrastive_loss(img_emb, txt_emb, logit_scale)

            if training:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping pour stabiliser l'entraînement
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def compute_retrieval_metrics(model, loader, device, k_list=(1, 5, 10)):
    """
    Calcule le Recall@K pour image→texte et texte→image sur le loader fourni.
    Utile pour monitorer la qualité de l'espace partagé en fin d'epoch.
    """
    model.eval()
    all_img_emb = []
    all_txt_emb = []

    with torch.no_grad():
        for images, captions in loader:
            images = images.to(device)
            img_e, txt_e, _ = model(images, captions)
            all_img_emb.append(img_e.cpu())
            all_txt_emb.append(txt_e.cpu())

    img_embs = torch.cat(all_img_emb)  # (N, D)
    txt_embs = torch.cat(all_txt_emb)  # (N, D)

    sim = img_embs @ txt_embs.T  # (N, N)

    results = {}
    for k in k_list:
        # Image → Texte
        topk_i2t = sim.topk(k, dim=1).indices                     # (N, k)
        gt_i2t   = torch.arange(len(img_embs)).unsqueeze(1)        # (N, 1)
        r_i2t    = (topk_i2t == gt_i2t).any(dim=1).float().mean().item()

        # Texte → Image
        topk_t2i = sim.T.topk(k, dim=1).indices
        gt_t2i   = torch.arange(len(txt_embs)).unsqueeze(1)
        r_t2i    = (topk_t2i == gt_t2i).any(dim=1).float().mean().item()

        results[f"R@{k}_i2t"] = r_i2t
        results[f"R@{k}_t2i"] = r_t2i

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device : {config.DEVICE}")
    device = torch.device(config.DEVICE)

    # ── Datasets & loaders ────────────────────────────────────────────────────
    print("Chargement des données …")
    train_ds, val_ds, _ = build_datasets(phase="phase1")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.PHASE1_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_phase1,
        drop_last=True,   # nécessaire pour que la matrice N×N soit carrée
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.PHASE1_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_phase1,
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    print("Initialisation du DualEncoder …")
    model = DualEncoder(
        embed_dim=config.EMBED_DIM,
        freeze_clip=True,   # poids ViT gelés, seules les projections sont entraînées
    ).to(device)

    # Paramètres entraînables uniquement
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Paramètres entraînables : {sum(p.numel() for p in trainable):,}")

    # ── Optimiseur & scheduler ────────────────────────────────────────────────
    optimizer = AdamW(
        trainable,
        lr=config.PHASE1_LR,
        weight_decay=config.PHASE1_WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.PHASE1_EPOCHS, eta_min=1e-6)

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, config.PHASE1_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, training=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, training=False)
        scheduler.step()

        # Métriques de retrieval (coûteux → seulement toutes les 5 epochs)
        ret_metrics = {}
        if epoch % 5 == 0 or epoch == config.PHASE1_EPOCHS:
            ret_metrics = compute_retrieval_metrics(model, val_loader, device)

        log = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            **ret_metrics,
        }
        history.append(log)

        ret_str = "  ".join(f"{k}={v:.3f}" for k, v in ret_metrics.items())
        print(
            f"Epoch {epoch:>3}/{config.PHASE1_EPOCHS} │ "
            f"train={train_loss:.4f}  val={val_loss:.4f}"
            + (f"  {ret_str}" if ret_str else "")
        )

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":   epoch,
                    "model":   model.state_dict(),
                    "optim":   optimizer.state_dict(),
                    "history": history,
                    "val_loss": val_loss,
                },
                config.PHASE1_CKPT,
            )
            print(f"  ✓ Meilleur checkpoint sauvegardé (val_loss={val_loss:.4f})")

    print(f"\nEntraînement Phase 1 terminé.  Checkpoint : {config.PHASE1_CKPT}")
    return history


if __name__ == "__main__":
    main()
