"""
train/train_phase2.py – Entraînement Phase 2 : Fine-tuning GPT-2 avec prefix visuel.

Pré-requis :
    Phase 1 terminée → checkpoint dans config.PHASE1_CKPT

Pipeline :
    1. Charge le tokenizer GPT-2 étendu avec nos tokens spéciaux
    2. Construit CaptioningModel (image encoder gelé + GPT-2 + mapper)
    3. Fine-tune sur Flickr8k avec causal LM loss (teacher forcing)
    4. Sauvegarde le meilleur checkpoint selon la val loss

Lancement :
    python train/train_phase2.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

import config
from dataset import build_datasets, collate_phase2
from decoder import CaptioningModel, build_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, training=True):
    model.train(training)
    total_loss = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, input_ids, attention_mask, labels in tqdm(
            loader, leave=False, desc="train" if training else "val"
        ):
            images         = images.to(device)
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)

            loss = model(images, input_ids, attention_mask, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)


def sample_generations(model, val_ds, tokenizer, device, n=4):
    """
    Génère quelques exemples sur le val set pour inspection qualitative.
    """
    model.eval()
    print("\n── Exemples de génération ──────────────────────────────────────")
    indices = torch.randperm(len(val_ds))[:n].tolist()
    for i in indices:
        image, _, _, _ = val_ds[i]
        image = image.unsqueeze(0).to(device)
        caption = model.generate_caption(image, tokenizer, num_beams=config.BEAM_SIZE)
        print(f"  [{i}] {caption}")
    print("────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device : {config.DEVICE}")
    device = torch.device(config.DEVICE)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("Chargement du tokenizer GPT-2 …")
    tokenizer = build_tokenizer()

    # ── Datasets & loaders ────────────────────────────────────────────────────
    print("Chargement des données …")
    train_ds, val_ds, _ = build_datasets(phase="phase2", tokenizer=tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.PHASE2_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_phase2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.PHASE2_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_phase2,
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    print("Initialisation du CaptioningModel …")
    model = CaptioningModel(
        phase1_ckpt=config.PHASE1_CKPT,
        prefix_len=10,
        mapper_variant="mlp",
        device=config.DEVICE,
    ).to(device)

    # Resize les embeddings GPT-2 si on a ajouté des tokens spéciaux
    model.gpt2.resize_token_embeddings(len(tokenizer))

    # Paramètres entraînables : mapper + GPT-2 (image encoder gelé)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Paramètres entraînables : {sum(p.numel() for p in trainable):,}")

    # ── Optimiseur & scheduler ────────────────────────────────────────────────
    # On utilise des taux d'apprentissage différents pour GPT-2 et le mapper
    mapper_params = list(model.mapper.parameters())
    gpt2_params   = list(model.gpt2.parameters())

    optimizer = AdamW(
        [
            {"params": mapper_params, "lr": config.PHASE2_LR * 5},   # mapper : LR × 5
            {"params": gpt2_params,   "lr": config.PHASE2_LR},        # GPT-2  : LR normal
        ],
        weight_decay=config.PHASE2_WEIGHT_DECAY,
    )

    # Warm restarts : T_0 epochs avant premier restart, T_mult=1 → tous les T_0
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-7)

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, config.PHASE2_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, training=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, training=False)
        scheduler.step()

        log = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "perplexity": torch.exp(torch.tensor(val_loss)).item(),
        }
        history.append(log)

        print(
            f"Epoch {epoch:>3}/{config.PHASE2_EPOCHS} │ "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"ppl={log['perplexity']:.2f}"
        )

        # Affichage qualitatif toutes les 5 epochs
        if epoch % 5 == 0:
            sample_generations(model, val_ds, tokenizer, device, n=3)

        # Sauvegarde
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
                config.PHASE2_CKPT,
            )
            print(f"  ✓ Meilleur checkpoint sauvegardé (val_loss={val_loss:.4f})")

    print(f"\nEntraînement Phase 2 terminé.  Checkpoint : {config.PHASE2_CKPT}")
    return history


if __name__ == "__main__":
    main()
