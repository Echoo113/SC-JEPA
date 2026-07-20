# ============================================================================
# Downstream Evaluation Entry Point
# ============================================================================
# Runs downstream anomaly prediction using frozen JEPA + soft codebook features.
# This file is the ONLY downstream entrypoint.
# ============================================================================

import argparse
import numpy as np
import torch

from downstream.utils import (
    load_dataset,
    set_seed,
    create_encoder,
    create_quantizer,
    load_pretrained,
)
from downstream.features import set_feature_models
from downstream.train_classifier import train_classifier
from downstream.evaluation import test_evaluation


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_f1, test_auc = [], []

    for run_id, seed in enumerate(args.seeds):
        print(f"\n{'='*60}")
        print(f"🚀 Run {run_id + 1}/{len(args.seeds)} | Seed = {seed}")
        print(f"{'='*60}")

        # --------------------------------------------------
        # Reproducibility
        # --------------------------------------------------
        set_seed(seed)

        # --------------------------------------------------
        # Load dataset (patch-level, preprocessed)
        # --------------------------------------------------
        _, dataset = load_dataset(args.data_path)

        # --------------------------------------------------
        # Build and load frozen JEPA + soft codebook models
        # --------------------------------------------------
        encoder = create_encoder(
            num_patches=args.num_patches,
            patch_len=args.patch_len,
            latent_dim=args.latent_dim,
        )
        quantizer = create_quantizer(
            num_codes=args.num_codes,
            embedding_dim=args.latent_dim,
        )

        encoder, quantizer = load_pretrained(
            encoder,
            quantizer,
            device=device,
            ckpt_path=args.ckpt,
        )

        # Register frozen models for feature extraction
        set_feature_models(encoder, quantizer, device)

        # --------------------------------------------------
        # Train downstream classifier (frozen features)
        # --------------------------------------------------
        classifier, best_thresh, _, _, test_loader = train_classifier(
            dataset,
            num_codes=args.num_codes,
            latent_dim=args.latent_dim,
            num_patches=args.num_patches,
            use_probs=args.use_probs,
            device=device,
            seed=seed,
            epochs=args.epochs,
            patience=args.patience,
        )

        # --------------------------------------------------
        # Final test evaluation (fixed threshold)
        # --------------------------------------------------
        _, f1, auc, _, _ = test_evaluation(
            classifier,
            test_loader,
            best_thresh,
            device=device,
        )

        test_f1.append(f1)
        test_auc.append(auc)

        print(f"✅ Test | F1={f1:.4f}, AUC={auc:.4f}")

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------
    print(f"\n{'='*60}")
    print("📊 Final Results")
    print(f"F1 : {np.mean(test_f1):.4f} ± {np.std(test_f1):.4f}")
    print(f"AUC: {np.mean(test_auc):.4f} ± {np.std(test_auc):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Downstream Anomaly Prediction")

    # Data / checkpoint
    parser.add_argument("--data_path", type=str, default="test_norm.npz")
    parser.add_argument("--ckpt", type=str, default="jepa_vqvae_best.pth")

    # Model config
    parser.add_argument("--num_codes", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_patches", type=int, default=5)
    parser.add_argument("--patch_len", type=int, default=20)

    # Downstream options
    parser.add_argument("--use_probs", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)

    # Experiment control
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    args = parser.parse_args()
    main(args)

    print("🔥 eval.py entered")

