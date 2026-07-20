# ============================================================================
# Downstream Classifier Training (CLEAN VERSION)
# ============================================================================
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from downstream.classifier import SimpleClassifier, FocalLoss
from downstream.features import extract_feature_maxpoolV
from downstream.evaluation import evaluate
from downstream.utils import data_split


def train_classifier(
    dataset,
    *,
    num_codes=128,
    latent_dim=256,
    num_patches=5,
    use_probs=True,
    batch_size=128,
    epochs=30,
    lr=2e-3,
    weight_decay=5e-5,
    patience=5,
    seed=42,
    device=None,
    save_ckpt=False,
    ckpt_path="best_classifier.pth",
):
    """
    Train window-level anomaly classifier with frozen JEPA + soft codebook features.

    Returns:
        classifier, best_threshold, train_loader, val_loader, test_loader
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Dataset split
    # ------------------------------------------------------------------
    train_set, val_set, test_set = data_split(
        dataset,
        train_ratio=0.60,
        val_ratio=0.20,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Classifier
    # ------------------------------------------------------------------
    in_dim = num_codes if use_probs else latent_dim
    classifier = SimpleClassifier(
        input_dim=in_dim,
        num_patches=num_patches,
    ).to(device)

    # ------------------------------------------------------------------
    # Focal loss (window-level)
    # ------------------------------------------------------------------
    train_labels = torch.stack([dataset.y_label[i] for i in train_set.indices])
    y_win = train_labels.max(dim=1).values
    neg, pos = (y_win == 0).sum(), (y_win == 1).sum()
    pos_weight = min((neg / (pos + 1e-6)).item() * 3.0, 15.0)

    criterion = FocalLoss(
        gamma=2.5,
        weight=torch.tensor([1.0, pos_weight], device=device),
    )

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_auc = -1.0
    best_thresh = 0.5
    wait = 0

    for epoch in range(1, epochs + 1):
        classifier.train()

        pbar = tqdm(train_loader, leave=False)
        for x_patch, y_label in pbar:
            x_patch = x_patch.to(device)
            y_label = y_label.to(device)

            feats = extract_feature_maxpoolV(
                x_patch,
                use_probs=use_probs,
            )

            logits = classifier(feats)
            y_win = y_label.max(dim=1).values
            loss = criterion(logits, y_win)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------
        val_thresh, _, _, val_auc, _ = evaluate(
            classifier,
            val_loader,
            return_metrics=True,
            verbose=False,
            device=device,
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_thresh = val_thresh
            wait = 0

            if save_ckpt:
                torch.save(
                    {
                        "state_dict": classifier.state_dict(),
                        "threshold": best_thresh,
                    },
                    ckpt_path,
                )
        else:
            wait += 1

        if wait >= patience:
            break

    # ------------------------------------------------------------------
    # IMPORTANT:
    # DO NOT torch.load classifier checkpoint here.
    # The classifier in memory is already the trained model.
    # ------------------------------------------------------------------

    return classifier, best_thresh, train_loader, val_loader, test_loader
