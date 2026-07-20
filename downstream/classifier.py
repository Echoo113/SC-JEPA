# ============================================================================
# Downstream classifier components.
# - FocalLoss for class imbalance.
# - SimpleClassifier for window-level prediction via patch aggregation.
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss with optional class weights (CrossEntropy-based)."""
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


class SimpleClassifier(nn.Module):
    """Classifier with attention pooling over patches."""
    def __init__(self, input_dim, num_patches=5):
        super().__init__()
        self.num_patches = num_patches
        self.input_dim = input_dim

        self.feature_enhance = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.attention = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B,P,D)
        B, P, D = x.shape

        x_enhanced = self.feature_enhance(x)  # (B,P,2D)

        attn_weights = torch.softmax(self.attention(x_enhanced), dim=1)  # (B,P,1)
        x_pooled = torch.sum(x_enhanced * attn_weights, dim=1)  # (B,2D)

        return self.classifier(x_pooled)  # (B,2)