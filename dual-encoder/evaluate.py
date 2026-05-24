"""
evaluate.py – Évaluation du modèle de captioning sur le test set.

Métriques calculées :
    • BLEU-1, BLEU-4  (précision des n-grammes)
    • METEOR           (recall + synonymes via WordNet)
    • ROUGE-L          (longest common subsequence)
    • CIDEr            (consensus-based, spécialement conçu pour le captioning)

Toutes ces métriques sont issues de pycocoevalcap (librairie MS-COCO officielle).

Lancement :
    python evaluate.py [--checkpoint checkpoints/decoder.pt] [--n 1000]
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import build_datasets, load_captions, make_splits, collate_phase2
from decoder import CaptioningModel, build_tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Wrappers pycocoevalcap
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    hypotheses: dict[str, list[str]],
    references: dict[str, list[str]],
) -> dict[str, float]:
    """
    hypotheses : { img_id: ["generated caption"] }
    references : { img_id: ["ref1", "ref2", "ref3", "ref4", "ref5"] }
    Retourne un dict { metric_name: score }.
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        print("⚠  pycocoevalcap non installé.  pip install pycocoevalcap")
        # Fallback : BLEU manuel avec nltk
        return _compute_bleu_fallback(hypotheses, references)

    scorers = [
        (Bleu(4),  ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), ["METEOR"]),
        (Rouge(),  ["ROUGE-L"]),
        (Cider(),  ["CIDEr"]),
    ]

    results = {}
    for scorer, names in scorers:
        score, _ = scorer.compute_score(references, hypotheses)
        if isinstance(score, list):
            for name, s in zip(names, score):
                results[name] = round(s, 4)
        else:
            results[names[0]] = round(score, 4)

    return results


def _compute_bleu_fallback(hypotheses, references) -> dict[str, float]:
    """Calcul BLEU via nltk si pycocoevalcap n'est pas disponible."""
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        import nltk
        nltk.download("punkt", quiet=True)
    except ImportError:
        return {"BLEU-1": -1, "BLEU-4": -1}

    sf = SmoothingFunction().method1
    all_refs, all_hyps = [], []
    for k in hypotheses:
        hyp = hypotheses[k][0].split()
        refs = [r.split() for r in references[k]]
        all_refs.append(refs)
        all_hyps.append(hyp)

    return {
        "BLEU-1": round(corpus_bleu(all_refs, all_hyps,
                                    weights=(1,0,0,0), smoothing_function=sf), 4),
        "BLEU-4": round(corpus_bleu(all_refs, all_hyps,
                                    weights=(.25,.25,.25,.25), smoothing_function=sf), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Génération sur le test set
# ─────────────────────────────────────────────────────────────────────────────

def generate_captions_for_split(
    model:      CaptioningModel,
    test_images: list[str],
    captions:    dict[str, list[str]],
    tokenizer,
    device,
    num_beams:  int = config.BEAM_SIZE,
    max_new:    int = config.MAX_NEW_TOKENS,
    n:          int | None = None,
) -> tuple[dict, dict]:
    """
    Génère une caption pour chaque image du test set.

    Retourne :
        hypotheses : { img_id: ["generated caption"] }
        references : { img_id: ["ref1", ..., "ref5"] }
    """
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(config.IMAGE_SIZE + 32),
        T.CenterCrop(config.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std= [0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    from PIL import Image as PilImage

    test_images_eval = test_images[:n] if n else test_images
    hypotheses, references = {}, {}

    model.eval()
    with torch.no_grad():
        for i, img_name in enumerate(tqdm(test_images_eval, desc="Génération")):
            img_path = os.path.join(config.IMAGES_DIR, img_name)
            img      = PilImage.open(img_path).convert("RGB")
            img_t    = transform(img).unsqueeze(0).to(device)

            caption  = model.generate_caption(
                img_t, tokenizer,
                max_new_tokens=max_new,
                num_beams=num_beams,
            )

            img_id = str(i)
            hypotheses[img_id] = [caption]
            references[img_id] = captions[img_name]

    return hypotheses, references


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Évaluation du modèle de captioning")
    parser.add_argument("--checkpoint", default=config.PHASE2_CKPT,
                        help="Chemin vers le checkpoint Phase 2")
    parser.add_argument("--n", type=int, default=None,
                        help="Nombre d'images à évaluer (None = tout le test set)")
    parser.add_argument("--beams", type=int, default=config.BEAM_SIZE)
    parser.add_argument("--output", default="results/eval_results.json",
                        help="Fichier JSON de sortie avec les scores")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device(config.DEVICE)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = build_tokenizer()

    # ── Modèle ────────────────────────────────────────────────────────────────
    print(f"Chargement du checkpoint : {args.checkpoint}")
    model = CaptioningModel(
        phase1_ckpt=config.PHASE1_CKPT,
        prefix_len=10,
        mapper_variant="mlp",
        device=config.DEVICE,
    ).to(device)
    model.gpt2.resize_token_embeddings(len(tokenizer))

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"  Epoch du checkpoint : {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")

    # ── Données test ──────────────────────────────────────────────────────────
    captions = load_captions(config.CAPTIONS_FILE)
    _, _, test_imgs = make_splits(captions)

    # ── Génération ────────────────────────────────────────────────────────────
    hypotheses, references = generate_captions_for_split(
        model, test_imgs, captions, tokenizer, device,
        num_beams=args.beams,
        n=args.n,
    )

    # ── Métriques ─────────────────────────────────────────────────────────────
    print("\nCalcul des métriques …")
    scores = compute_metrics(hypotheses, references)

    print("\n══ Résultats ══════════════════════════════")
    for metric, value in scores.items():
        print(f"  {metric:<12} {value:.4f}")
    print("═══════════════════════════════════════════\n")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    output_data = {
        "checkpoint": args.checkpoint,
        "n_images":   len(hypotheses),
        "beam_size":  args.beams,
        "scores":     scores,
        "examples":   [
            {
                "img_id":     k,
                "hypothesis": hypotheses[k][0],
                "references": references[k],
            }
            for k in list(hypotheses.keys())[:10]  # 10 premiers pour inspection
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Résultats sauvegardés dans {args.output}")
    return scores


if __name__ == "__main__":
    main()
