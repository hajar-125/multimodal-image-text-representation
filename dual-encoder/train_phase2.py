"""
models/decoder.py – Phase 2 : Image-conditioned GPT-2 caption decoder.

Architecture :
    • Image encoder (gelé)  → features (B, clip_dim)
    • Projection linéaire   → (B, gpt2_hidden)     ← seule couche entraînable
    • Reshape en prefix     → (B, prefix_len, gpt2_hidden)
    • GPT-2 reçoit ce prefix + les tokens texte → génère la suite

Inspiration BLIP / ClipCap :
    On n'injecte pas via le cross-attention (complexité) mais via un
    "visual prefix" : l'embedding image est transformé en une séquence
    de tokens visuels fictifs passés en tête du contexte GPT-2.
    Ainsi GPT-2 peut être fine-tuné avec le standard causal LM loss.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from dual_encoder import load_image_encoder


# ─────────────────────────────────────────────────────────────────────────────
# MLP Mapper : features image → prefix tokens GPT-2
# ─────────────────────────────────────────────────────────────────────────────

class ImagePrefixMapper(nn.Module):
    """
    Transforme l'embedding image (clip_dim) en une séquence de
    `prefix_len` tokens pseudo-visuels de dimension `gpt2_hidden`.

    Deux variantes :
        - "linear"  : couche linéaire simple + reshape (rapide, moins expressif)
        - "mlp"     : MLP à 2 couches + GELU + reshape
    """

    def __init__(
        self,
        clip_dim:   int = config.EMBED_DIM,
        gpt2_dim:   int = 768,      # gpt2-small hidden size
        prefix_len: int = 10,       # nombre de tokens visuels préfixés
        variant:    str = "mlp",
    ):
        super().__init__()
        self.prefix_len = prefix_len
        out_dim = gpt2_dim * prefix_len

        if variant == "linear":
            self.mapper = nn.Linear(clip_dim, out_dim)
        else:  # mlp
            self.mapper = nn.Sequential(
                nn.Linear(clip_dim, clip_dim * 2),
                nn.GELU(),
                nn.Linear(clip_dim * 2, out_dim),
            )
        self.norm = nn.LayerNorm(gpt2_dim)

    def forward(self, img_emb: torch.Tensor) -> torch.Tensor:
        """
        img_emb : (B, clip_dim)
        Retourne : (B, prefix_len, gpt2_dim)
        """
        B = img_emb.size(0)
        out = self.mapper(img_emb)                      # (B, gpt2_dim * prefix_len)
        out = out.view(B, self.prefix_len, -1)          # (B, prefix_len, gpt2_dim)
        out = self.norm(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CaptioningModel
# ─────────────────────────────────────────────────────────────────────────────

class CaptioningModel(nn.Module):
    """
    Modèle de génération de captions (Phase 2).

    Paramètres
    ----------
    phase1_ckpt : str
        Chemin vers le checkpoint de la Phase 1 (dual encoder).
    prefix_len  : int
        Nombre de tokens visuels préfixés injectés dans GPT-2.
    mapper_variant : "linear" | "mlp"

    Flux forward (entraînement)
    ---------------------------
    images → image_encoder(gelé) → image_proj(gelé) → ImagePrefixMapper
           → prefix_embeds (B, prefix_len, 768)
    tokens → GPT-2 token embeddings → text_embeds (B, T, 768)
    concat [prefix_embeds | text_embeds] → GPT-2 transformer → logits
    loss   = cross_entropy(logits[:, prefix_len:], labels)
             (on ne calcule la loss que sur la partie texte)
    """

    def __init__(
        self,
        phase1_ckpt:    str  = config.PHASE1_CKPT,
        prefix_len:     int  = 10,
        mapper_variant: str  = "mlp",
        device:         str  = config.DEVICE,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        self.device     = device

        # ── 1. Image encoder gelé (depuis Phase 1) ────────────────────────────
        print(f"[CaptioningModel] Chargement de l'encodeur image depuis {phase1_ckpt}")
        self.image_encoder, self.image_proj = load_image_encoder(phase1_ckpt, device)
        for p in self.image_encoder.parameters(): p.requires_grad = False
        for p in self.image_proj.parameters():    p.requires_grad = False

        # ── 2. GPT-2 ──────────────────────────────────────────────────────────
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.GPT2_MODEL_NAME)
        
        # 🚨 AJOUTEZ CES DEUX LIGNES ICI POUR FIXER L'INDEXERROR 🚨
        from train_phase2 import build_tokenizer
        tokenizer = build_tokenizer()
        self.gpt2.resize_token_embeddings(len(tokenizer))
        
        gpt2_dim  = self.gpt2.config.n_embd  # 768 pour gpt2-small

        # ── 3. Mapper image → prefix tokens ───────────────────────────────────
        self.mapper = ImagePrefixMapper(
            clip_dim=config.EMBED_DIM,
            gpt2_dim=gpt2_dim,
            prefix_len=prefix_len,
            variant=mapper_variant,
        )
    # ── Extraction des features image (avec gradient désactivé) ──────────────

    @torch.no_grad()
    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        images : (B, 3, H, W)
        Retourne : (B, embed_dim) – embeddings normalisés
        """
        import torch.nn.functional as F
        feats = self.image_encoder(images)
        proj  = self.image_proj(feats.float())
        return F.normalize(proj, dim=-1)

    # ── Forward (entraînement) ────────────────────────────────────────────────

    def forward(
        self,
        images:         torch.Tensor,   # (B, 3, H, W)
        input_ids:      torch.Tensor,   # (B, T)
        attention_mask: torch.Tensor,   # (B, T)
        labels:         torch.Tensor,   # (B, T)  -100 sur les tokens à ignorer
    ) -> torch.Tensor:
        """
        Retourne la loss de language model sur la partie texte uniquement.
        """
        B, T = input_ids.shape

        # ── Embeddings image ──────────────────────────────────────────────────
        img_emb      = self.extract_image_features(images)   # (B, embed_dim)
        prefix_emb   = self.mapper(img_emb)                  # (B, prefix_len, gpt2_dim)

        # ── Embeddings texte ──────────────────────────────────────────────────
        token_emb = self.gpt2.transformer.wte(input_ids)     # (B, T, gpt2_dim)

        # ── Concaténation : [prefix | texte] ──────────────────────────────────
        inputs_embeds = torch.cat([prefix_emb, token_emb], dim=1)  # (B, prefix_len+T, gpt2_dim)

        # Attention mask étendu : les prefix tokens sont toujours visibles
        prefix_mask   = torch.ones(B, self.prefix_len, device=images.device)
        full_mask     = torch.cat([prefix_mask, attention_mask], dim=1)  # (B, prefix_len+T)

        # Labels étendus : -100 pour les prefix tokens (pas de loss sur eux)
        prefix_labels = torch.full((B, self.prefix_len), -100, device=images.device)
        full_labels   = torch.cat([prefix_labels, labels], dim=1)  # (B, prefix_len+T)

        # ── Passage dans GPT-2 ────────────────────────────────────────────────
        outputs = self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )
        return outputs.loss

    # ── Génération (inférence) ────────────────────────────────────────────────

    @torch.no_grad()
    def generate_caption(
        self,
        image:          torch.Tensor,   # (1, 3, H, W) ou (3, H, W)
        tokenizer:      GPT2Tokenizer,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        num_beams:      int = config.BEAM_SIZE,
        temperature:    float = 1.0,
        do_sample:      bool  = False,
    ) -> str:
        """
        Génère une caption pour une seule image.

        Stratégies disponibles :
            - Beam search   : do_sample=False, num_beams > 1 (par défaut)
            - Greedy        : do_sample=False, num_beams=1
            - Sampling      : do_sample=True,  temperature < 1 → plus déterministe
        """
        self.eval()
        device = next(self.parameters()).device

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        # Prefix visuel
        img_emb    = self.extract_image_features(image)   # (1, embed_dim)
        prefix_emb = self.mapper(img_emb)                 # (1, prefix_len, gpt2_dim)

        # Token de début
        bos_id = tokenizer.encode(config.BOS_TOKEN, add_special_tokens=False)
        bos_id = torch.tensor([bos_id], device=device)   # (1, 1)
        bos_emb = self.gpt2.transformer.wte(bos_id)       # (1, 1, gpt2_dim)

        # Inputs initiaux = [prefix | BOS]
        inputs_embeds = torch.cat([prefix_emb, bos_emb], dim=1)  # (1, prefix_len+1, gpt2_dim)

        # Génération
        output_ids = self.gpt2.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )
        # Décodage : on ignore le BOS
        generated = output_ids[0]
        caption = tokenizer.decode(generated, skip_special_tokens=True)
        return caption.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Préparation du tokenizer GPT-2 (ajout des tokens spéciaux)
# ─────────────────────────────────────────────────────────────────────────────

def build_tokenizer() -> GPT2Tokenizer:
    """
    Charge GPT2Tokenizer et ajoute les tokens spéciaux de début/fin.
    Le padding token est mis à <|endoftext|> (convention GPT-2).
    """
    tokenizer = GPT2Tokenizer.from_pretrained(config.GPT2_MODEL_NAME)
    tokenizer.add_special_tokens({"bos_token": config.BOS_TOKEN})
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Boucle d'entraînement complète (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader

# On importe les outils de gestion des données depuis votre fichier dataset.py
# Note : Ajustez le nom des importations si vos classes s'appellent différemment dans dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# Boucle d'entraînement complète (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────
import os
from torch.utils.data import DataLoader

# On importe les fonctions exactes de votre dataset.py
from dataset import build_datasets, collate_phase2

def train():
    print(f"[{config.DEVICE}] Démarrage de la Phase 2 (Image Captioning)...")
    device = config.DEVICE
    
    # 1. Initialiser le tokenizer et le modèle
    tokenizer = build_tokenizer()
    model = CaptioningModel(
        phase1_ckpt=config.PHASE1_CKPT,
        prefix_len=10,
        mapper_variant="mlp",
        device=device
    ).to(device)
    
    # 2. Configurer l'optimizer (Uniquement sur le mapper de projection)
    optimizer = torch.optim.AdamW(model.mapper.parameters(), lr=1e-4)
    
    # 3. Charger les jeux de données et DataLoaders via vos fonctions dédiées
    print("Chargement et découpage des données Flickr8k...")
    train_ds, val_ds, _ = build_datasets(phase="phase2", tokenizer=tokenizer)
    
    # Création des loaders avec votre fonction de collation personnalisée
    train_loader = DataLoader(
        train_ds,
        batch_size=32, # Vous pouvez baisser à 16 ou 8 si vous saturez la mémoire (Out of Memory)
        shuffle=True,
        collate_fn=collate_phase2
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_phase2
    )
    
    # S'assurer que le dossier pour sauvegarder vos checkpoints existe
    os.makedirs("checkpoints", exist_ok=True)
    
    # 4. Boucle d'entraînement principale
    epochs = getattr(config, "EPOCHS", 5) # Récupère la valeur dans config.py ou met 5 par défaut
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        print(f"\n--- Époque {epoch + 1}/{epochs} ---")
        
        for step, batch in enumerate(train_loader):
            # Récupération du tuple structuré par collate_phase2
            images, input_ids, attention_mask, labels = batch
            
            # Envoi des tenseurs sur le bon périphérique (CPU / GPU / MPS)
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Reset des gradients
            optimizer.zero_grad()
            
            # Forward pass : calcul de la loss
            loss = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Backward pass & mise à jour des poids
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Affichage de la progression toutes les 50 étapes
            if step % 50 == 0:
                print(f"Étape [{step}/{len(train_loader)}] | Loss Courante: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"-> Fin de l'Époque {epoch + 1} | Loss Moyenne Entraînement: {avg_train_loss:.4f}")
        
        # 5. Validation à la fin de chaque époque
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, input_ids, attention_mask, labels = batch
                images = images.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                loss = model(images, input_ids, attention_mask, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"-> Loss Moyenne Validation: {avg_val_loss:.4f}")
            
        # 6. Sauvegarde du checkpoint
        checkpoint_path = f"checkpoints/captioning_phase2_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"💾 Checkpoint sauvegardé : {checkpoint_path}")

    print("\nL'entraînement de la Phase 2 est terminé ! 🚀")

if __name__ == "__main__":
    train()