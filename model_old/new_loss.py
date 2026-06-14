import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ================================
# 1. ADVANCED LOSS FUNCTIONS
# ================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance
    Better than weighted CrossEntropy for extreme imbalance
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Can be tensor of weights for each class
        if isinstance(self.alpha, list) or isinstance(self.alpha, np.ndarray):
            self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Regression Loss for score prediction
    Better than standard classification for ordered labels
    """
    def __init__(self, num_classes=21):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes]
        targets: [batch_size] (scores from 0-10, representing 0-10)
        """
        batch_size = targets.size(0)
        device = targets.device
        
        # Create ordinal targets
        ordinal_targets = torch.zeros(batch_size, self.num_classes - 1, device=device)

        score_idx = targets.long().clamp(0, self.num_classes - 1)
        for i in range(batch_size):
            # score_idx = targets[i].long()
            if score_idx[i] > 0:
                ordinal_targets[i, :score_idx[i]] = 1.0
        
        # Use first n-1 logits for ordinal regression
        ordinal_logits = logits[:, :-1]
        
        # Binary cross entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(ordinal_logits, ordinal_targets)
        return loss


class KLFocalLoss(nn.Module):
    """
    KL + Focal Loss with optional per-class alpha weighting.
    Works with soft targets (distributions).
    """
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1):
        """
        Args:
            alpha (float or Tensor): 
                - float: global scaling factor
                - Tensor [num_classes]: per-class weights
            gamma (float): focusing parameter for hard examples
            smoothing (float): label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, soft_targets):
        num_classes = soft_targets.size(1)

        # Apply label smoothing
        smooth_targets = soft_targets * (1 - self.smoothing) + self.smoothing / num_classes

        # Log-softmax for KL
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Expected probability assigned to soft target distribution
        pt = (probs * smooth_targets).sum(dim=-1)  # [B]

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma  # [B]

        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Expand per-class alpha into batch
                # shape: [B, num_classes]
                alpha_t = (self.alpha.to(logits.device) * smooth_targets).sum(dim=-1)
            else:
                # Scalar alpha
                alpha_t = torch.full_like(pt, float(self.alpha))
            focal_weight = focal_weight * alpha_t  # [B]

        # KL divergence
        kl_loss = F.kl_div(log_probs, smooth_targets, reduction='none').sum(dim=-1)  # [B]
        loss = (focal_weight * kl_loss).mean()

        return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss using Effective Number of Samples
    More sophisticated than simple inverse weighting
    """
    def __init__(self, effective_weights: torch.Tensor, beta=0.9999, gamma=2.0, loss_type='focal'):
        super().__init__()

        self.effective_weights = effective_weights
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
    
    def forward(self, logits, targets):
        self.effective_weights = self.effective_weights.to(logits.device)
        
        if self.loss_type == 'focal':
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            weights = self.effective_weights[targets]
            focal_loss = weights * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()
        else:
            return F.cross_entropy(logits, targets, weight=self.effective_weights, reduction='mean')

class ImbalanceAwareLoss(nn.Module):
    """
    Combines multiple loss functions for imbalanced data
    """
    def __init__(self, class_counts, effective_weights, gamma_focal=2.0, 
                 lambda_kl=0.4, lambda_ordinal=0.3, lambda_focal=0.3):
        super().__init__()
        
        # Initialize different loss functions
        print('INIT LOSS WITH CLASS COUNTS:', class_counts)
        print('Length of class counts:', len(class_counts))

        if not isinstance(effective_weights, torch.Tensor):
            effective_weights = torch.tensor(effective_weights, dtype=torch.float32)
        self.effective_weights = effective_weights
        self.focal_loss = ClassBalancedLoss(effective_weights=self.effective_weights, gamma=gamma_focal, loss_type='focal')
        self.ordinal_loss = OrdinalRegressionLoss(num_classes=len(class_counts))
        self.distribution_loss = KLFocalLoss(alpha=class_counts, gamma=gamma_focal)

        # Loss weights
        self.lambda_kl = lambda_kl
        self.lambda_ordinal = lambda_ordinal
        self.lambda_focal = lambda_focal


    
    def forward(self, logits, soft_targets, hard_targets):
        """
        logits: model predictions [batch, num_classes]
        soft_targets: soft target distribution [batch, num_classes]
        hard_targets: hard target indices [batch]
        """
        # KL divergence loss (your existing)
        log_probs = F.log_softmax(logits, dim=-1)
        kl_loss = F.kl_div(log_probs, soft_targets, reduction='mean')
        
        # Focal loss for hard targets
        focal_loss = self.focal_loss(logits, hard_targets.long())
        
        # Ordinal regression loss
        ordinal_loss = self.ordinal_loss(logits, hard_targets.long())
        
        # Combined loss
        total_loss = (self.lambda_kl * kl_loss + 
                     self.lambda_focal * focal_loss + 
                     self.lambda_ordinal * ordinal_loss)
        
        return total_loss, {
            'kl_loss': kl_loss.item(),
            'focal_loss': focal_loss.item(),
            'ordinal_loss': ordinal_loss.item(),
            'total_loss': total_loss.item()
        }