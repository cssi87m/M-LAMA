"""
Loss Functions for ESL Speaking Grading Model

Implements:
- MAE and band loss for |err|<=margin
- Focal Regression Loss: Focus on hard samples
- Ranking Loss: Preserve score ordering
- Distribution-Aware Loss: Penalize edge↔middle errors
- Soft target generation with edge-aware smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Focal Regression Loss
# ============================================================================

def compute_focal_regression_loss(pred_scores, true_scores, weights,
                                   gamma=2.0, beta=1.0):
    """
    Focal Loss for regression - Focus on hard samples

    Loss = |y - ŷ|^gamma * smooth_l1(y, ŷ)

    Args:
        pred_scores: [B] - predicted scores (0-10)
        true_scores: [B] - true scores (0-10)
        weights: [B] - class weights
        gamma: Focusing parameter (default 2.0) - higher focuses more on hard samples
        beta: SmoothL1 beta parameter

    Returns:
        Scalar loss value
    """
    # Compute absolute error
    abs_error = torch.abs(pred_scores - true_scores)

    # Focal weight: error càng lớn → weight càng cao
    # FIXED: Don't divide by 10 - makes weights too small!
    # Use abs_error directly, capped at 1.0 to prevent extreme values
    focal_weight = torch.pow(torch.clamp(abs_error / 3.0, max=1.0), gamma)

    # Base loss: SmoothL1
    smooth_l1 = F.smooth_l1_loss(pred_scores, true_scores, reduction='none', beta=beta)

    # Combine
    focal_loss = focal_weight * smooth_l1

    # Apply class weights
    weighted_loss = (focal_loss * weights).sum() / weights.sum()

    return weighted_loss


# ============================================================================
# MAE Loss
# ============================================================================

def compute_mae_loss(pred_scores, true_scores, weights):
    """
    Weighted MAE loss to optimize absolute error directly.

    Args:
        pred_scores: [B] - predicted scores
        true_scores: [B] - true scores
        weights: [B] - class weights

    Returns:
        Scalar loss value
    """
    abs_error = torch.abs(pred_scores - true_scores)
    weighted_loss = (abs_error * weights).sum() / weights.sum()
    return weighted_loss

# ============================================================================
# Ranking Loss
# ============================================================================

def compute_ranking_loss(logits, true_scores, weights, margin=0.5):
    """
    Ranking Loss - Ensure correct score ordering

    Idea: If score_i > score_j, then pred_i should be > pred_j

    Args:
        logits: [B, 21] - model outputs
        true_scores: [B] - true scores
        weights: [B] - class weights
        margin: Minimum margin between scores

    Returns:
        Scalar loss value
    """
    probs = F.softmax(logits, dim=-1)
    score_bins = torch.linspace(0, 10, steps=21, device=probs.device)
    pred_scores = (probs * score_bins).sum(dim=-1)

    B = pred_scores.size(0)

    # Pairwise comparisons
    # Shape: [B, B]
    true_diff = true_scores.unsqueeze(1) - true_scores.unsqueeze(0)  # true_i - true_j
    pred_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)  # pred_i - pred_j

    # Ranking loss: max(0, margin - sign(true_diff) * pred_diff)
    # If true_i > true_j (true_diff > 0), we want pred_i > pred_j (pred_diff > 0)
    sign_true = torch.sign(true_diff)
    ranking_loss = F.relu(margin - sign_true * pred_diff)

    # Only compute for pairs with different scores
    # FIXED: Changed from >0.5 to >0.4 to catch pairs with 0.5 score difference
    # (since scores are in 0.5 increments: 0.0, 0.5, 1.0, ...)
    valid_pairs = (torch.abs(true_diff) > 0.4).float()
    ranking_loss = ranking_loss * valid_pairs

    # Weight pairs by class weights
    pair_weights = weights.unsqueeze(1) * weights.unsqueeze(0)
    denom = (pair_weights * valid_pairs).sum()
    eps = 1e-8
    weighted_ranking = (ranking_loss * pair_weights).sum() / torch.clamp(denom, min=eps)

    return weighted_ranking


# ============================================================================
# Distribution-Aware Loss
# ============================================================================

def compute_distribution_aware_loss(logits, true_scores, weights,
                                     edge_threshold=3.5, edge_penalty=2.5, mid_penalty=2.5):
    """
    Symmetric Edge Protection Loss

    Penalizes:
    1. Middle predictions for edge samples (edge→mid)
    2. Edge predictions for middle samples (mid→edge) - NEW!

    Edge samples: scores <= edge_threshold or >= (10 - edge_threshold)
    Middle samples: edge_threshold < scores < (10 - edge_threshold)

    Args:
        logits: [B, 21] - model outputs
        true_scores: [B] - true scores (0-10)
        weights: [B] - class weights
        edge_threshold: Scores considered "edge" (default 3.5)
        edge_penalty: Penalty multiplier for edge→mid (default 2.5)
        mid_penalty: Penalty multiplier for mid→edge (default 2.5) - NEW!

    Returns:
        Scalar loss value
    """
    probs = F.softmax(logits, dim=-1)
    score_bins = torch.linspace(0, 10, steps=21, device=probs.device)
    pred_scores = (probs * score_bins).sum(dim=-1)

    # Identify edge and middle samples
    is_edge = (true_scores <= edge_threshold) | (true_scores >= (10 - edge_threshold))
    is_mid = (true_scores > edge_threshold) & (true_scores < (10 - edge_threshold))

    # Identify edge and middle predictions
    is_edge_pred = (pred_scores <= edge_threshold) | (pred_scores >= (10 - edge_threshold))
    is_middle_pred = (pred_scores > edge_threshold) & (pred_scores < (10 - edge_threshold))

    # Penalty masks
    penalty_mask_edge_to_mid = is_edge & is_middle_pred  # Edge sample → Mid prediction
    penalty_mask_mid_to_edge = is_mid & is_edge_pred     # Mid sample → Edge prediction (NEW!)

    # Base loss: MSE
    mse_loss = F.mse_loss(pred_scores, true_scores, reduction='none')

    # Apply penalties
    loss = mse_loss.clone()
    loss[penalty_mask_edge_to_mid] *= edge_penalty  # Penalize edge→mid
    loss[penalty_mask_mid_to_edge] *= mid_penalty   # Penalize mid→edge (NEW!)

    # Apply class weights
    weighted_loss = (loss * weights).sum() / weights.sum()

    return weighted_loss


# ============================================================================
# Band Loss for |err| <= margin
# ============================================================================

def compute_band_loss(pred_scores, true_scores, weights, margin=1.0):
    """
    Penalize predictions outside a margin around ground truth.

    Args:
        pred_scores: [B] - predicted scores
        true_scores: [B] - true scores
        weights: [B] - class weights
        margin: Allowed absolute error (default 1.0)

    Returns:
        Scalar loss value
    """
    abs_error = torch.abs(pred_scores - true_scores)
    band = F.relu(abs_error - margin)
    band_loss = band ** 2
    weighted_loss = (band_loss * weights).sum() / weights.sum()
    return weighted_loss

# ============================================================================
# Soft Target Generation
# ============================================================================

def create_soft_targets(scores, std=0.3, edge_threshold=3.5, edge_std_multiplier=0.5):
    """
    Generate soft targets with less smoothing for edge scores

    Args:
        scores: [B] - true scores (0-10)
        std: Base smoothing strength (default 0.3)
        edge_threshold: Scores considered "edge"
        edge_std_multiplier: Reduce smoothing for edge (0.5 = 50% smoothing)

    Returns:
        Soft labels: [B, 21] - probability distributions over score bins
    """
    scores_np = scores.cpu().numpy()
    B = scores_np.shape[0]
    soft_labels = np.zeros((B, 21), dtype=np.float32)

    for i, score in enumerate(scores_np):
        target_bin = int(round(score * 2))
        target_bin = np.clip(target_bin, 0, 20)

        # Determine if edge score
        is_edge = (score <= edge_threshold) or (score >= (10 - edge_threshold))

        # Adjust smoothing
        effective_std = std * edge_std_multiplier if is_edge else std

        # Create smoothed distribution
        one_hot = np.zeros(21)
        one_hot[target_bin] = 1.0
        smoothed = one_hot.copy()

        if target_bin > 0:
            smoothed[target_bin - 1] = effective_std
            smoothed[target_bin] -= effective_std
        if target_bin < 20:
            smoothed[target_bin + 1] = effective_std
            smoothed[target_bin] -= effective_std

        smoothed = np.maximum(smoothed, 0.0)
        smoothed = smoothed / smoothed.sum()
        soft_labels[i] = smoothed

    return torch.from_numpy(soft_labels).to(scores.device)


# ============================================================================
# NEW: KL Divergence and Entropy Loss (for sharp predictions)
# ============================================================================

def create_one_hot_targets(scores, num_bins=21):
    """
    Create one-hot targets for KL divergence loss

    Args:
        scores: [B] - true scores (0-10)
        num_bins: Number of score bins (default 21 for 0.5 increments)

    Returns:
        one_hot: [B, 21] - one-hot target distributions
    """
    B = scores.size(0)
    device = scores.device

    # Convert continuous scores to bin indices
    bin_indices = torch.round(scores * 2).long()  # 0→0, 0.5→1, 1.0→2, ..., 10→20
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    # Create one-hot encoding
    one_hot = torch.zeros(B, num_bins, device=device)
    one_hot.scatter_(1, bin_indices.unsqueeze(1), 1.0)

    return one_hot


def compute_kl_divergence_loss(logits, target_distributions, weights):
    """
    Compute KL divergence between predicted and target distributions

    KL(target || predicted) encourages predicted distribution to match target

    Args:
        logits: [B, 21] - model output logits
        target_distributions: [B, 21] - target probability distributions (one-hot or smoothed)
        weights: [B] - class weights

    Returns:
        Weighted KL divergence loss (scalar)
    """
    # Predicted distribution
    log_pred = F.log_softmax(logits, dim=-1)  # [B, 21]

    # KL divergence: sum(target * log(target / pred))
    # = sum(target * (log(target) - log(pred)))
    # For numerical stability with one-hot targets (where target can be 0):
    # KL = -sum(target * log_pred) + sum(target * log_target)
    # Since target is one-hot, sum(target * log_target) = 0
    # So KL = -sum(target * log_pred) = CrossEntropy

    kl_loss = -(target_distributions * log_pred).sum(dim=-1)  # [B]

    # Apply class weights
    weighted_loss = (kl_loss * weights).sum() / weights.sum()

    return weighted_loss


def compute_entropy_penalty(logits, weights):
    """
    Compute entropy of predicted distributions to penalize flat/uncertain predictions

    Lower entropy = more peaked/confident predictions (desired)
    Higher entropy = more uniform/uncertain predictions (penalized)

    Args:
        logits: [B, 21] - model output logits
        weights: [B] - class weights

    Returns:
        Weighted entropy penalty (scalar)
    """
    # Predicted distribution
    probs = F.softmax(logits, dim=-1)  # [B, 21]
    log_probs = F.log_softmax(logits, dim=-1)

    # Entropy: H(p) = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)  # [B]

    # Higher entropy = bad (more spread out)
    # Penalize high entropy
    weighted_loss = (entropy * weights).sum() / weights.sum()

    return weighted_loss


# ============================================================================
# Combined Loss Function
# ============================================================================

def compute_combined_loss(outputs, true_scores, weights, config):
    """
    Combined loss function with multiple components

    Args:
        outputs: Dict with 'expected_score', 'logits', 'probs'
        true_scores: [B] - ground truth scores
        weights: [B] - class weights
        config: TrainingConfig object

    Returns:
        total_loss: Combined weighted loss
        loss_dict: Dictionary of individual loss components
    """
    pred_scores = outputs['expected_score']  # [B]
    logits = outputs['logits']              # [B, 21]

    loss_dict = {}

    # 1. MAE Loss (Primary)
    mae_loss = compute_mae_loss(pred_scores, true_scores, weights)
    loss_dict['mae'] = mae_loss.item()

    # 2. KL Divergence with One-Hot Targets (NEW - Encourage exact predictions)
    if hasattr(config, 'lambda_kl') and config.lambda_kl > 0:
        one_hot_targets = create_one_hot_targets(true_scores, num_bins=21)
        kl_loss = compute_kl_divergence_loss(logits, one_hot_targets, weights)
        loss_dict['kl'] = kl_loss.item()
    else:
        kl_loss = torch.tensor(0.0, device=pred_scores.device)
        loss_dict['kl'] = 0.0

    # 3. Entropy Penalty (NEW - Discourage flat distributions)
    if hasattr(config, 'lambda_entropy') and config.lambda_entropy > 0:
        entropy_loss = compute_entropy_penalty(logits, weights)
        loss_dict['entropy'] = entropy_loss.item()
    else:
        entropy_loss = torch.tensor(0.0, device=pred_scores.device)
        loss_dict['entropy'] = 0.0

    # 4. Focal Loss (Optional)
    if config.lambda_focal > 0:
        focal_loss = compute_focal_regression_loss(
            pred_scores, true_scores, weights,
            gamma=config.focal_gamma,
            beta=config.focal_beta
        )
        loss_dict['focal'] = focal_loss.item()
    else:
        focal_loss = torch.tensor(0.0, device=pred_scores.device)
        loss_dict['focal'] = 0.0

    # 5. Ranking Loss (Optional)
    if config.lambda_ranking > 0:
        ranking_loss = compute_ranking_loss(
            logits, true_scores, weights,
            margin=config.ranking_margin
        )
        loss_dict['ranking'] = ranking_loss.item()
    else:
        ranking_loss = torch.tensor(0.0, device=pred_scores.device)
        loss_dict['ranking'] = 0.0

    # 6. Distribution-Aware Loss (Optional - reduced/disabled)
    if config.lambda_dist > 0:
        dist_loss = compute_distribution_aware_loss(
            logits, true_scores, weights,
            edge_threshold=config.edge_threshold,
            edge_penalty=config.edge_penalty,
            mid_penalty=config.mid_penalty
        )
        loss_dict['dist'] = dist_loss.item()
    else:
        dist_loss = torch.tensor(0.0, device=pred_scores.device)
        loss_dict['dist'] = 0.0

    # 7. Band Loss (Optional)
    if config.lambda_band > 0:
        band_loss = compute_band_loss(
            pred_scores, true_scores, weights,
            margin=config.band_margin
        )
        loss_dict['band'] = band_loss.item()
    else:
        band_loss = torch.tensor(0.0, device=pred_scores.device)
        loss_dict['band'] = 0.0

    # Combine all losses
    total_loss = (
        config.lambda_mae * mae_loss +
        (config.lambda_kl if hasattr(config, 'lambda_kl') else 0) * kl_loss +
        (config.lambda_entropy if hasattr(config, 'lambda_entropy') else 0) * entropy_loss +
        config.lambda_focal * focal_loss +
        config.lambda_ranking * ranking_loss +
        config.lambda_dist * dist_loss +
        config.lambda_band * band_loss
    )

    loss_dict['total_loss'] = total_loss.item()

    return total_loss, loss_dict
# ============================================================================
# Legacy Loss Functions (from old code)
# ============================================================================

def compute_kl_loss(logits, soft_targets, weights):
    """
    KL Divergence Loss (from old code)

    Args:
        logits: [B, 21] - model outputs
        soft_targets: [B, 21] - soft target distributions
        weights: [B] - class weights

    Returns:
        Scalar loss
    """
    log_probs = F.log_softmax(logits, dim=-1)
    kl_loss_per_sample = F.kl_div(log_probs, soft_targets, reduction='none').sum(dim=-1)
    weighted_loss = (kl_loss_per_sample * weights).sum() / weights.sum()
    return weighted_loss


def compute_smoothl1_loss(pred_scores, true_scores, weights, beta=1.0):
    """
    Smooth L1 Loss (from old code)

    Args:
        pred_scores: [B] - predicted scores
        true_scores: [B] - true scores
        weights: [B] - class weights
        beta: SmoothL1 beta parameter

    Returns:
        Scalar loss
    """
    smoothl1_per_sample = F.smooth_l1_loss(pred_scores, true_scores, reduction='none', beta=beta)
    weighted_loss = (smoothl1_per_sample * weights).sum() / weights.sum()
    return weighted_loss


if __name__ == "__main__":
    # Test loss functions
    print("✓ Losses module loaded successfully")

    # Test data
    batch_size = 4
    logits = torch.randn(batch_size, 21)
    true_scores = torch.tensor([3.0, 5.0, 7.5, 9.0])
    weights = torch.ones(batch_size)

    # Test Focal Loss
    pred_scores = torch.tensor([3.5, 5.5, 7.0, 8.0])
    focal_loss = compute_focal_regression_loss(pred_scores, true_scores, weights)
    print(f"Focal Loss: {focal_loss.item():.4f}")

    # Test Ranking Loss
    ranking_loss = compute_ranking_loss(logits, true_scores, weights)
    print(f"Ranking Loss: {ranking_loss.item():.4f}")

    # Test Distribution-Aware Loss
    dist_loss = compute_distribution_aware_loss(logits, true_scores, weights)
    print(f"Distribution-Aware Loss: {dist_loss.item():.4f}")

    # Test Soft Targets
    soft_targets = create_soft_targets(true_scores)
    print(f"Soft targets shape: {soft_targets.shape}")
    print(f"First target (score=3.0): {soft_targets[0, 4:8]}")  # bins around 3.0
