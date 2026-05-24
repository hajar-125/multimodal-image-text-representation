"""
models/dual_encoder.py – Phase 1 : Dual Encoder contrastif (CLIP-style).

Architecture :
    • Image encoder  : ViT-B/32 via open_clip (pré-entraîné CLIP)
    • Text encoder   : projection linéaire sur les embeddings CLIP text
    • Les deux branches projettent vers EMBED_DIM dimensions
    • Loss           : InfoNCE symétrique (image→text + text→image)

Pourquoi open_clip ?
    On part d'un modèle ViT déjà aligné image-texte pour accélérer la
    convergence sur Flickr8k (petit dataset).  La phase 1 affine cet
    espace partagé sur les 6000 images d'entraînement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Projection head
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """
    Tête de projection MLP : embed_dim → hidden → out_dim + LayerNorm.
    Utilisée pour adapter les features brutes vers l'espace partagé.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or out_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


# ─────────────────────────────────────────────────────────────────────────────
# Dual Encoder
# ─────────────────────────────────────────────────────────────────────────────

class DualEncoder(nn.Module):
    """
    Encodeur dual image-texte inspiré de CLIP / BLIP.

    Paramètres
    ----------
    embed_dim : int
        Dimension de l'espace partagé.
    freeze_clip : bool
        Si True, les poids CLIP sont gelés → seules les projection heads
        sont entraînées.  Recommandé avec Flickr8k (petit dataset) pour
        éviter le surapprentissage.
    """

    def __init__(
        self,
        embed_dim:   int  = config.EMBED_DIM,
        freeze_clip: bool = True,
    ):
        super().__init__()

        # ── Chargement du modèle CLIP de base ─────────────────────────────────
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME,
            pretrained=config.CLIP_PRETRAINED,
        )
        self.tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)

        # On garde séparément l'encodeur image et l'encodeur texte
        self.image_encoder = clip_model.visual            # ViT-B/32 visual trunk
        self.text_encoder  = clip_model                   # on utilisera encode_text

        # Dimension de sortie du ViT-B/32 CLIP
        clip_embed_dim = clip_model.visual.output_dim     # 512 pour ViT-B/32

        # ── Projection heads ──────────────────────────────────────────────────
        self.image_proj = ProjectionHead(clip_embed_dim, embed_dim)
        self.text_proj  = ProjectionHead(clip_embed_dim, embed_dim)

        # ── Température apprise ───────────────────────────────────────────────
        self.log_temp = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / config.PHASE1_TEMPERATURE))
        )

        # ── Gel optionnel des poids CLIP ──────────────────────────────────────
        if freeze_clip:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    # ── Encodage image ────────────────────────────────────────────────────────

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images : (B, 3, H, W) – tenseurs normalisés avec les stats CLIP
        Retourne des embeddings L2-normalisés de shape (B, embed_dim).
        """
        with torch.set_grad_enabled(self.image_encoder.training):
            feats = self.image_encoder(images)   # (B, clip_embed_dim)
        proj  = self.image_proj(feats)           # (B, embed_dim)
        return F.normalize(proj, dim=-1)

    # ── Encodage texte ────────────────────────────────────────────────────────

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        texts : liste de strings (batch)
        Retourne des embeddings L2-normalisés de shape (B, embed_dim).
        Le tokenizer CLIP est appliqué ici.
        """
        tokens = self.tokenizer(texts).to(next(self.parameters()).device)
        with torch.set_grad_enabled(self.text_encoder.training):
            feats = self.text_encoder.encode_text(tokens)   # (B, clip_embed_dim)
        proj = self.text_proj(feats.float())                 # (B, embed_dim)
        return F.normalize(proj, dim=-1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        texts:  list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retourne (image_embeds, text_embeds, logit_scale).
        """
        img_emb  = self.encode_image(images)
        txt_emb  = self.encode_text(texts)
        logit_scale = self.log_temp.exp()
        return img_emb, txt_emb, logit_scale


# ─────────────────────────────────────────────────────────────────────────────
# Loss InfoNCE symétrique
# ─────────────────────────────────────────────────────────────────────────────

def contrastive_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Perte InfoNCE symétrique (image→texte + texte→image) / 2.

    Paramètres
    ----------
    img_emb, txt_emb : (B, D) – embeddings L2-normalisés
    logit_scale       : scalaire appris (= 1/température)

    La matrice de similarité cosinus est multipliée par logit_scale avant
    le softmax.  Les labels sont les indices diagonaux (paires positives).
    """
    B = img_emb.size(0)
    device = img_emb.device

    # Matrice de similarité (B, B)
    sim = logit_scale * img_emb @ txt_emb.T   # (B, B)

    labels = torch.arange(B, device=device)

    # Cross-entropy dans les deux sens puis moyenne
    loss_i2t = F.cross_entropy(sim,   labels)   # image vers texte
    loss_t2i = F.cross_entropy(sim.T, labels)   # texte vers image

    return (loss_i2t + loss_t2i) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Utility : récupérer uniquement l'encodeur image (pour la phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def load_image_encoder(checkpoint_path: str, device: str = config.DEVICE):
    """
    Charge un DualEncoder depuis un checkpoint et retourne uniquement
    l'image_encoder + image_proj (ce dont on a besoin en phase 2).
    Les poids sont gelés.
    """
    model = DualEncoder().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # Gel total
    for p in model.parameters():
        p.requires_grad = False

    return model.image_encoder, model.image_proj
