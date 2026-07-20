# ============================================================================
# Evaluation utilities for downstream anomaly prediction.
# - Threshold selection via F-beta.
# - Validation evaluation and test reporting.
# ============================================================================

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm import tqdm

from downstream.features import extract_feature_maxpoolV

# Module-level configuration
USE_PROBS_MAXPOOL = True


def set_evaluation_backend(use_probs_maxpool=True, device_=None):
    """
    Configure evaluation backend.
    """
    global USE_PROBS_MAXPOOL
    USE_PROBS_MAXPOOL = use_probs_maxpool


def _best_threshold(y_true, y_prob, beta=1.0, n=200):
    """
    Select threshold by maximizing F-beta over a grid.
    """
    if y_prob.size == 0:
        return 0.5, 0.0

    if y_prob.std() < 1e-6:
        fallback_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        best, best_t = -1, 0.5
        b2 = beta * beta

        for t in fallback_thresholds:
            y_pred = (y_prob > t).astype(int)
            p, r, _, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            fbeta = (1 + b2) * p * r / (b2 * p + r + 1e-12)
            if fbeta > best:
                best, best_t = fbeta, t
        return best_t, best

    ts = np.linspace(y_prob.min(), y_prob.max(), n)
    best, best_t = -1, 0.5
    b2 = beta * beta

    for t in ts:
        y_pred = (y_prob > t).astype(int)
        p, r, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        fbeta = (1 + b2) * p * r / (b2 * p + r + 1e-12)
        if fbeta > best:
            best, best_t = fbeta, t

    return best_t, best


def evaluate(classifier, loader, return_metrics=False, verbose=True, device=None):
    """
    Window-level evaluation with threshold optimization.
    
    Evaluates classifier on window-level anomaly prediction.
    All variables are aggregated via variable-wise max pooling in feature extraction.
    Labels are at window level (max over patches if needed).
    
    Args:
        classifier: Trained SimpleClassifier
        loader: DataLoader with batches
        return_metrics: Return (threshold, f1, (p, r, f1), auc, acc)
        verbose: Print results
        device: torch device to move data to
    
    Returns:
        If return_metrics=False: best_threshold
        If return_metrics=True: (best_threshold, f1, (p, r, f1), auc, accuracy)
        
    Note:
        - F1 score is window-level binary classification metric
        - y_label shape: (B,P) → converted to window-level (B,) via max over P
        - Feature extraction: variable-wise max pooling aggregates all V variables
        - Threshold selected via F-beta (beta=1.0) on this loader
    """
    classifier.eval()
    y_true_all, y_prob_all = [], []

    for x_patch, y_label in loader:
        if device is not None:
            x_patch = x_patch.to(device)
            y_label = y_label.to(device)
        
        # Extract features: (B, P, L, V) → (B, P, K) via variable-wise max pooling
        feats = extract_feature_maxpoolV(x_patch, use_probs=USE_PROBS_MAXPOOL)
        logits = classifier(feats)  # (B, 2) window-level logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # (B,) anomaly probabilities

        # Convert to window-level labels
        if y_label.dim() == 2:  # (B, P)
            y_win = y_label.max(dim=1).values  # (B,) - if any patch is anomalous, window is anomalous
        else:  # Already (B,)
            y_win = y_label

        y_true_all.extend(y_win.cpu().tolist())
        y_prob_all.extend(probs.cpu().tolist())

    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)

    # Select best threshold by maximizing F1 (F-beta with beta=1.0)
    best_thresh, _ = _best_threshold(y_true, y_prob, beta=1.0, n=200)

    # Compute window-level metrics using binary classification with best threshold
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, (y_prob > best_thresh).astype(int),
        average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    y_pred_best = (y_prob > best_thresh).astype(int)
    accuracy = (y_true == y_pred_best).mean()

    if verbose:
        print(
            f"✅ Window-level F1={f1:.4f} @Thresh={best_thresh:.4f} | "
            f"P={p:.4f}, R={r:.4f}, AUC={auc:.4f}, Acc={accuracy:.4f}"
        )

    if return_metrics:
        return best_thresh, f1, (p, r, f1), auc, accuracy
    return best_thresh


@torch.no_grad()
def test_evaluation(classifier, test_loader, val_threshold, device=None):
    """
    Final test evaluation with window-level F1 metric.
    
    Uses a fixed threshold from validation set to ensure no data leakage.
    Computes window-level anomaly prediction metrics.
    All variables aggregated via variable-wise max pooling in feature extraction.
    
    Args:
        classifier: Trained SimpleClassifier
        test_loader: Test DataLoader with batches
        val_threshold: Threshold selected on validation set
        device: torch device to move data to
    
    Returns:
        (val_threshold, f1, auc, precision, recall)
        - F1: Window-level binary classification F1 score
        - AUC: Area under ROC curve at window level
        - Precision, Recall: Window-level metrics
        
    Note:
        - F1 is computed as binary classification at window level
        - y_label shape: (B,P) → converted to window-level (B,) via max over P
        - Feature extraction: variable-wise max pooling aggregates all V variables
        - Threshold is fixed (from validation), not re-optimized on test set
    """
    classifier.eval()
    y_true_all, y_prob_all = [], []

    for x_patch, y_label in tqdm(test_loader, desc="Test evaluation"):
        if device is not None:
            x_patch = x_patch.to(device)
            y_label = y_label.to(device)
        
        # Extract features: (B, P, L, V) → (B, P, K) via variable-wise max pooling
        feats = extract_feature_maxpoolV(x_patch, use_probs=USE_PROBS_MAXPOOL)
        logits = classifier(feats)  # (B, 2) window-level logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # (B,) anomaly probabilities

        # Convert to window-level labels
        if y_label.dim() == 2:  # (B, P)
            y_win = y_label.max(dim=1).values  # (B,) - if any patch is anomalous, window is anomalous
        else:  # Already (B,)
            y_win = y_label

        y_true_all.extend(y_win.cpu().tolist())
        y_prob_all.extend(probs.cpu().tolist())

    y_true = np.array(y_true_all)
    y_prob = np.array(y_prob_all)

    if y_prob.size == 0:
        print("⚠️ Test loader yielded no samples.")
        return val_threshold, 0.0, 0.0, 0.0, 0.0

    # Apply fixed validation threshold
    y_pred = (y_prob > val_threshold).astype(int)

    # Compute window-level metrics using binary classification
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

    print(f"\n📊 Test Results (Window-level metrics, all variables aggregated):")
    print(f"   F1={f1:.4f}, AUC={auc:.4f}")
    print(f"   Precision={p:.4f}, Recall={r:.4f}")
    print(f"   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    spec = tn/(tn+fp) if (tn+fp) > 0 else 0
    acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) > 0 else 0
    print(f"   Specificity={spec:.4f}, Accuracy={acc:.4f}")

    return val_threshold, f1, auc, p, r
