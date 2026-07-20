"""
Minimal preprocessing pipeline (dataset-agnostic).

Important:
- Pretraining is fully self-supervised and does NOT use labels.
- Training labels are set to zeros only to keep the same data structure
  as the test set (x_patches / y_patches + x_label / y_label).
- Constant variables are identified from the training data ONLY,
  and the same variable mask is applied to both training and test data
  to ensure a consistent feature space.
"""

import numpy as np


# -------------------------------------------------
# Remove constant variables (train-based mask)
# -------------------------------------------------

def build_keep_mask(train_data, var_threshold=1e-8):
    """
    train_data: (T_train, V)
    Returns:
      keep_mask: (V,) boolean
    """
    var = np.var(train_data, axis=0)
    return var > var_threshold


# -------------------------------------------------
# Sliding window + patch extraction
# -------------------------------------------------

def split_to_patches(data, label,
                     history_len=100,
                     future_len=100,
                     patch_len=20):
    """
    data : (T, V)
    label: (T,)
    Returns:
      X_patches, X_label, Y_patches, Y_label
    """
    stride = history_len
    P = history_len // patch_len

    Xs, Ys, Xl, Yl = [], [], [], []

    for i in range(0, len(data) - history_len - future_len + 1, stride):
        x = data[i:i+history_len].reshape(P, patch_len, -1)
        y = data[i+history_len:i+history_len+future_len].reshape(P, patch_len, -1)

        lx = label[i:i+history_len].reshape(P, patch_len).any(axis=1)
        ly = label[i+history_len:i+history_len+future_len].reshape(P, patch_len).any(axis=1)

        Xs.append(x); Ys.append(y)
        Xl.append(lx.astype(int)); Yl.append(ly.astype(int))

    return (np.array(Xs), np.array(Xl),
            np.array(Ys), np.array(Yl))


# -------------------------------------------------
# Per-variable normalization
# -------------------------------------------------

def normalize(x, eps=1e-6):
    mean = x.mean(axis=(0,1,2), keepdims=True)
    std  = x.std(axis=(0,1,2), keepdims=True) + eps
    return (x - mean) / std


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------

def preprocess(train_data, test_data, test_label):
    # Training labels are NOT used in pretraining
    train_label = np.zeros(len(train_data), dtype=int)

    # 1) Remove constant variables (mask from train, applied to both)
    keep_mask = build_keep_mask(train_data)
    train_data = train_data[:, keep_mask]
    test_data  = test_data[:, keep_mask]

    # 2) Windowing + patching
    Xtr, Xtr_l, Ytr, Ytr_l = split_to_patches(train_data, train_label)
    Xte, Xte_l, Yte, Yte_l = split_to_patches(test_data, test_label)

    # 3) Normalization (per-variable)
    Xtr = normalize(Xtr); Ytr = normalize(Ytr)
    Xte = normalize(Xte); Yte = normalize(Yte)

    return {
        "train": (Xtr, Ytr, Xtr_l, Ytr_l),
        "test":  (Xte, Yte, Xte_l, Yte_l),
        "keep_mask": keep_mask
    }
