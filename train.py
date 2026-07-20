# train.py
"""
Training entry point for JEPA with soft codebook.

This script serves as a minimal executable interface for reproducing
the training procedure reported in the paper.
"""

import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# =========================
# Dataset & utilities
# =========================
from data.datasets import ChannelIndependentDataset
from data.utils import (
    instance_normalize,          # per-sample normalization (RevIN-style)
    reverse_instance_normalize,  # invert instance normalization
    update_ema,                  # EMA update for target networks
    coarse_scale_to_patch,       # downscale future to coarse temporal patches
)

# =========================
# Loss functions
# =========================
from engine.losses import (
    kl_loss_fine,        # fine-grained JEPA KL loss
    kl_loss_coarse,      # coarse-scale JEPA KL loss
    mse_alignment_loss,  # latent regression (predictor → target latent)
    vq_losses,           # codebook + commitment losses
    entropy_losses,      # sample-level & batch-level entropy regularization
)

from engine.validate import validate_model

# =========================
# Model components
# =========================
from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import Quantizer
from models.predictor import TransformerPredictor, CoarsePredictor


def train():
    """
    Main training routine for JEPA with soft codebook and
    fine- and coarse-scale future prediction.
    """

    # -------------------------------------------------
    # Device
    # -------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {DEVICE}")

    # -------------------------------------------------
    # Training hyperparameters
    # -------------------------------------------------
    NUM_EPOCHS = 1  # Run 1 epoch for demonstration
    BATCH_SIZE = 128
    LEARNING_RATE = 5e-4
    EMA_DECAY = 0.996  # momentum encoder decay

    # -------------------------------------------------
    # Loss weights
    # -------------------------------------------------
    KL_FINE_WEIGHT = 1.0
    KL_COARSE_WEIGHT = 0.5
    MSE_WEIGHT = 0.1
    BETA = 1.0                     # soft codebook loss weight
    COMMITMENT_WEIGHT = 0.25       # commitment loss
    ENTROPY_SAMPLE_WEIGHT = 0.005  # per-sample entropy regularization
    ENTROPY_BATCH_WEIGHT = 0.01    # batch-level entropy regularization
    PRED_TEMP = 0.8                # temperature for KL prediction loss
    RECON_WEIGHT_START = 0.5       # reconstruction loss annealing start
    RECON_WEIGHT_END = 0.1         # reconstruction loss annealing end

    # -------------------------------------------------
    # Load preprocessed patch data
    # -------------------------------------------------
    DATA_DIR = "./"
    print(f"[INFO] Loading train_norm.npz from {DATA_DIR}...")
    train_np = torch.from_numpy(
        __import__("numpy").load(os.path.join(DATA_DIR, "train_norm.npz"))["x_patches"]
    ).float()
    print(f"[INFO] Data shape: {train_np.shape}")

    # Extract dimensions from data
    _, NUM_PATCHES, PATCH_LEN, IN_CHANNELS = train_np.shape
    print(f"[INFO] Extracted: NUM_PATCHES={NUM_PATCHES}, PATCH_LEN={PATCH_LEN}, IN_CHANNELS={IN_CHANNELS}")

    # -------------------------------------------------
    # Train / validation split 
    # -------------------------------------------------
    split_idx = int(len(train_np) * 0.9)
    idx_train = torch.arange(split_idx)
    idx_val = torch.arange(split_idx, len(train_np))

    train_ds = ChannelIndependentDataset(train_np[idx_train])
    val_ds = ChannelIndependentDataset(train_np[idx_val])

    # Data loaders
    

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print(f"[INFO] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    IN_CHANNELS = 1
    # -------------------------------------------------
    # Model initialization
    # -------------------------------------------------
    LATENT_DIM = 256
    NUM_CODES = 128
    H_DIM = 256
    RES_H_DIM = 128
    N_RES_LAYERS = 2
    NHEAD = 4
    NUM_TRANS_LAYERS = 2

    print(f"[INFO] Initializing models...")
    encoder = Encoder(NUM_PATCHES, PATCH_LEN, LATENT_DIM, H_DIM, RES_H_DIM, N_RES_LAYERS, NHEAD, NUM_TRANS_LAYERS, IN_CHANNELS).to(DEVICE)
    quantizer = Quantizer(NUM_CODES, LATENT_DIM).to(DEVICE)

    # Fine-scale predictor (Transformer JEPA)
    predictor = TransformerPredictor(num_codes=NUM_CODES, nhead=NHEAD, num_layers=NUM_TRANS_LAYERS, hidden_dim=128, num_patches=NUM_PATCHES, latent_dim=LATENT_DIM).to(DEVICE)

    # Coarse-scale predictor (lower temporal resolution)
    coarse_predictor = CoarsePredictor(num_codes=NUM_CODES, nhead=NHEAD, num_layers=NUM_TRANS_LAYERS, hidden_dim=128, num_patches=NUM_PATCHES, latent_dim=LATENT_DIM).to(DEVICE)

    # Decoder for reconstruction (soft codebook branch)
    decoder = Decoder(LATENT_DIM, IN_CHANNELS, PATCH_LEN).to(DEVICE)

    # -------------------------------------------------
    # EMA target networks (no gradients)
    # -------------------------------------------------
    encoder_tgt = copy.deepcopy(encoder).eval()
    quantizer_tgt = copy.deepcopy(quantizer).eval()

    for p in encoder_tgt.parameters():
        p.requires_grad = False
    for p in quantizer_tgt.parameters():
        p.requires_grad = False

    # -------------------------------------------------
    # Optimizer (online networks only)
    # -------------------------------------------------
    optimizer = optim.Adam(
        list(encoder.parameters())
        + list(quantizer.parameters())
        + list(predictor.parameters())
        + list(coarse_predictor.parameters())
        + list(decoder.parameters()),
        lr=LEARNING_RATE,
    )

    # =================================================
    # Training loop
    # =================================================
    for epoch in range(NUM_EPOCHS):
        print(f"\n[EPOCH {epoch+1}/{NUM_EPOCHS}]")
        encoder.train()
        quantizer.train()
        predictor.train()
        coarse_predictor.train()
        decoder.train()

        batch_count = 0
        for x_past, x_future in train_loader:
            batch_count += 1
            
            # -------------------------------------------------
            # Anneal reconstruction loss weight
            # -------------------------------------------------
            progress = batch_count / len(train_loader)
            recon_weight = RECON_WEIGHT_START - (RECON_WEIGHT_START - RECON_WEIGHT_END) * progress
            x_past_norm, mean, std = instance_normalize(x_past)
            x_future_norm, _, _ = instance_normalize(x_future)

            # Coarse-scale future patches (temporal downsampling)
            x_future_coarse = coarse_scale_to_patch(
                x_future, add_patch_dim=True
            )

            x_past_norm = x_past_norm.to(DEVICE)
            x_future_norm = x_future_norm.to(DEVICE)
            x_future_coarse = x_future_coarse.to(DEVICE)
            mean, std = mean.to(DEVICE), std.to(DEVICE)

            # -------------------------------------------------
            # Encode + quantize past context
            # -------------------------------------------------
            h_past = encoder(x_past_norm)
            p_past, z_q_past = quantizer(h_past)

            # -------------------------------------------------
            # Reconstruction loss (soft codebook branch)
            # -------------------------------------------------
            x_recon = decoder(z_q_past)
            x_recon = reverse_instance_normalize(x_recon, mean, std)

            # Target is original past window
            x_target = x_past.permute(0, 1, 3, 2).to(DEVICE)

            loss_recon = torch.nn.functional.mse_loss(
                x_recon, x_target
            )

            # -------------------------------------------------
            # Fine-scale JEPA prediction
            # -------------------------------------------------
            logits_pred, z_pred = predictor(p_past)

            # Target future (EMA encoder + EMA quantizer)
            h_future = encoder_tgt(x_future_norm)
            p_future, z_q_future = quantizer_tgt(h_future)

            loss_kl = kl_loss_fine(
                logits_pred, p_future, PRED_TEMP
            )

            # -------------------------------------------------
            # Coarse-scale JEPA prediction
            # -------------------------------------------------
            logits_coarse, _ = coarse_predictor(p_past)

            h_future_c = encoder_tgt(
                instance_normalize(x_future_coarse)[0]
            )
            p_future_c, _ = quantizer_tgt(h_future_c)

            loss_kl_c = kl_loss_coarse(
                logits_coarse, p_future_c, PRED_TEMP
            )

            # -------------------------------------------------
            # Latent alignment (predictor regression)
            # -------------------------------------------------
            loss_mse = mse_alignment_loss(
                z_pred, z_q_future
            )

            # -------------------------------------------------
            # Soft codebook losses
            # -------------------------------------------------
            loss_q, loss_commit = vq_losses(
                h_past, z_q_past
            )

            # -------------------------------------------------
            # Entropy regularization (avoid code collapse)
            # -------------------------------------------------
            loss_ent_s, loss_ent_b = entropy_losses(
                p_past
            )

            # -------------------------------------------------
            # Total loss
            # -------------------------------------------------
            loss = (
                KL_FINE_WEIGHT * loss_kl
                + KL_COARSE_WEIGHT * loss_kl_c
                + MSE_WEIGHT * loss_mse
                + BETA * loss_q
                + COMMITMENT_WEIGHT * loss_commit
                + ENTROPY_SAMPLE_WEIGHT * loss_ent_s
                + ENTROPY_BATCH_WEIGHT * loss_ent_b
                + recon_weight * loss_recon
            )

            # -------------------------------------------------
            # Optimization step
            # -------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -------------------------------------------------
            # EMA update (momentum target networks)
            # -------------------------------------------------
            update_ema(encoder, encoder_tgt, EMA_DECAY)
            update_ema(quantizer, quantizer_tgt, EMA_DECAY)

            if batch_count % max(1, len(train_loader) // 3) == 0 or batch_count == len(train_loader):
                print(f"  Batch {batch_count}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"[EPOCH {epoch+1}] Complete")


def main():
    train()


if __name__ == "__main__":
    main()
