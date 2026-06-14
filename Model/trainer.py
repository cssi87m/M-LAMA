"""
Training logic for ESL Speaking Grading Model

Key features:
- MAE as primary metric (instead of MSE)
- Combined loss: Focal + Ranking + Distribution-Aware
- Gradient accumulation and mixed precision training
- Best model tracking based on validation MAE
- Comprehensive wandb logging for experiment tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import os
from collections import defaultdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from dataloader import (
    ESLDatasetByCandidatesWithAudio,
    StratifiedScoreSampler,
    get_collate_fn_bycandidates_with_audio
)
from losses import compute_combined_loss
from utils import (
    clean_dataframe_bycandidates,
    get_class_counts_from_dataframe,
    get_effective_number_weights,
    maybe_empty_cache
)


class ESLTrainerByCandidatesWithAudio:
    """
    Trainer for ESL Grading Model with Audio

    Features:
    - MAE as primary evaluation metric
    - Combined Focal + Ranking + Distribution-Aware loss
    - Gradient accumulation for larger effective batch size
    - Mixed precision training (AMP)
    - Best model tracking based on validation MAE
    """

    def __init__(self,
                 model,
                 tokenizer,
                 audio_processor,
                 optimizer,
                 scheduler,
                 config,
                 device='cuda'):
        """
        Args:
            model: ESLGradingModelByCandidatesWithAudio instance
            tokenizer: HuggingFace tokenizer
            audio_processor: Wav2Vec2Processor
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            config: Config object with all settings
            device: Device for training
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Training config
        self.batch_size = config.training.batch_size
        self.eval_batch_size = config.training.eval_batch_size if config.training.eval_batch_size is not None else config.training.batch_size
        self.accumulation_steps = config.training.accumulation_steps
        self.epochs = config.training.epochs
        self.max_grad_norm = config.training.max_grad_norm

        # Data config
        self.criteria = config.data.criteria
        self.num_workers = config.data.num_workers
        self.prefetch_factor = config.data.prefetch_factor

        # Mixed precision
        self.scaler = GradScaler()

        # Best model tracking (MAE-based)
        self.best_val_mae = float('inf')
        self.best_state_dict = None

        # Logging
        self.log_every_n_steps = config.logging.log_every_n_steps
        self.wandb_enabled = config.logging.wandb_enabled and WANDB_AVAILABLE

    def _prepare_data(self):
        """
        Load and prepare train/val/test datasets

        Key changes:
        - Uses criteria='final' instead of 'grammar'
        - Creates class weights using Effective Number method
        """
        print("=" * 80)
        print("Loading datasets...")

        # Load dataframes
        train_df = pd.read_csv(self.config.data.train_path)
        val_df = pd.read_csv(self.config.data.val_path)
        test_df = pd.read_csv(self.config.data.test_path)

        # ví dụ lấy 100 dòng đầu mỗi file
        # train_df = pd.read_csv(self.config.data.train_path, nrows=50)
        # val_df   = pd.read_csv(self.config.data.val_path,   nrows=50)
        # test_df  = pd.read_csv(self.config.data.test_path,  nrows=50)


        print(f"Raw data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # Clean dataframes
        # train_df = clean_dataframe_bycandidates(
        #     train_df,
        #     remove_low_content=False,
        #     filter_scores=True,
        #     criteria=self.criteria
        # )
        # val_df = clean_dataframe_bycandidates(
        #     val_df,
        #     remove_low_content=False,
        #     filter_scores=True,
        #     criteria=self.criteria
        # )
        # test_df = clean_dataframe_bycandidates(
        #     test_df,
        #     remove_low_content=False,
        #     filter_scores=True,
        #     criteria=self.criteria
        # )

        print(f"After cleaning: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # Create datasets
        train_dataset = ESLDatasetByCandidatesWithAudio(
            train_df,
            criteria=self.criteria,
            audio_processor=self.audio_processor,
            encoder_type=self.config.model.audio_encoder_type,  # NEW
            num_chunks=self.config.audio.num_chunks,
            chunk_length_sec=self.config.audio.chunk_length_sec,
            separate_question_response=self.config.model.use_question_encoder  # STEP 3
        )
        # Use eval_num_chunks for val/test if specified, otherwise use num_chunks
        eval_num_chunks = self.config.audio.eval_num_chunks if self.config.audio.eval_num_chunks is not None else self.config.audio.num_chunks

        val_dataset = ESLDatasetByCandidatesWithAudio(
            val_df,
            criteria=self.criteria,
            audio_processor=self.audio_processor,
            encoder_type=self.config.model.audio_encoder_type,  # NEW
            num_chunks=eval_num_chunks,  # Use eval_num_chunks for validation
            chunk_length_sec=self.config.audio.chunk_length_sec,
            separate_question_response=self.config.model.use_question_encoder  # STEP 3
        )
        test_dataset = ESLDatasetByCandidatesWithAudio(
            test_df,
            criteria=self.criteria,
            audio_processor=self.audio_processor,
            encoder_type=self.config.model.audio_encoder_type,  # NEW
            num_chunks=eval_num_chunks,  # Use eval_num_chunks for test
            chunk_length_sec=self.config.audio.chunk_length_sec,
            separate_question_response=self.config.model.use_question_encoder  # STEP 3
        )

        # Compute class weights
        class_bins = [i * 0.5 for i in range(21)]  # 0, 0.5, 1.0, ..., 10.0
        class_counts = get_class_counts_from_dataframe(
            train_df,
            class_bins,
            criteria=self.criteria
        )
        self.loss_weights = get_effective_number_weights(
            class_counts,
            beta=self.config.data.class_weight_beta
        ).to(self.device)

        print(f"\nClass weights (top 5): {self.loss_weights[:5].tolist()}")
        print(f"Class weights (bottom 5): {self.loss_weights[-5:].tolist()}")

        # Create sampler (StratifiedScore)
        train_sampler = StratifiedScoreSampler(
            train_dataset,
            edge_threshold=self.config.training.edge_threshold,
            edge_ratio=self.config.data.edge_ratio
        )

        # Collate function
        collate_fn = get_collate_fn_bycandidates_with_audio(
            self.tokenizer,
            max_length=self.config.data.max_length,
            max_audio_chunks=self.config.audio.max_audio_chunks,
            max_waveform_len=self.config.audio.max_waveform_len,
            separate_tokenize=self.config.model.use_question_encoder  # STEP 3
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.eval_batch_size,  # Use eval_batch_size for validation
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,  # Use eval_batch_size for test
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True
        )

        print("✓ Data preparation complete")
        print("=" * 80)

    def train(self, start_epoch: int = 0):
        """
        Main training loop

        Key features:
        - Track MAE as primary metric (instead of MSE)
        - Save best model based on validation MAE
        - Use combined loss (Focal + Ranking + Distribution-Aware)
        """
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80)

        self._prepare_data()

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model.encoder, 'gradient_checkpointing_enable'):
            self.model.encoder.gradient_checkpointing_enable()

        # Initial validation before training (to check baseline from checkpoint)
        # print("\n" + "=" * 80)
        # print("INITIAL VALIDATION (Baseline from checkpoint)")
        # print("=" * 80)
        # torch.cuda.empty_cache()
        # gc.collect()

        # initial_val_metrics = self.validate()
        # initial_mae = initial_val_metrics['mae']
        # print(f"\nInitial Baseline MAE: {initial_mae:.4f}")
        # print(f"Initial Acc|err|<=1.0: {initial_val_metrics['acc_err_le_1_0']*100:.2f}%")
        # print("=" * 80)

        # # Log initial validation to wandb
        # if self.wandb_enabled:
        #     wandb.log({
        #         'epoch': 0,
        #         'val/mae': initial_mae,
        #         'val/mse': initial_val_metrics['mse'],
        #         'val/weighted_mse': initial_val_metrics['weighted_mse'],
        #         'val/mae_edge': initial_val_metrics['mae_edge'],
        #         'val/mae_mid': initial_val_metrics['mae_mid'],
        #         'val/acc_err_le_1_0': initial_val_metrics['acc_err_le_1_0'],
        #     }, step=0)

        # # Set initial baseline as best if better than default
        # if initial_mae < self.best_val_mae:
        #     self.best_val_mae = initial_mae
        #     print(f"\n✓ Initial checkpoint MAE ({initial_mae:.4f}) set as baseline to beat")

        # Clear VRAM before starting training
        torch.cuda.empty_cache()
        gc.collect()

        global_step = 0

        for epoch in range(start_epoch, self.epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"{'='*80}")

            # Training phase
            self.model.train()
            train_losses = defaultdict(list)

            pbar = tqdm(self.train_loader, desc=f"Training")

            for batch_idx, batch in enumerate(pbar):
                # Move to device
                audio = batch['audio'].to(self.device)
                true_scores = batch['score'].to(self.device)

                # STEP 3: Conditional input based on use_question_encoder
                if self.config.model.use_question_encoder:
                    question_input_ids = batch['question_input_ids'].to(self.device)
                    question_attention_mask = batch['question_attention_mask'].to(self.device)
                    response_input_ids = batch['response_input_ids'].to(self.device)
                    response_attention_mask = batch['response_attention_mask'].to(self.device)

                    # Forward pass with mixed precision
                    with autocast():
                        outputs = self.model(
                            question_input_ids=question_input_ids,
                            question_attention_mask=question_attention_mask,
                            response_input_ids=response_input_ids,
                            response_attention_mask=response_attention_mask,
                            audio=audio
                        )
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    # Forward pass with mixed precision
                    with autocast():
                        outputs = self.model(input_ids, attention_mask, audio)

                # Compute target indexes and weights
                target_indexes = (true_scores * 2).long().clamp(0, 20)
                weights = self.loss_weights[target_indexes]

                # Combined loss
                loss, loss_dict = compute_combined_loss(
                    outputs, true_scores, weights, self.config.training
                )

                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps

                # Backward pass
                self.scaler.scale(loss).backward()

                # Update weights every accumulation_steps
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()

                    global_step += 1

                    if self.wandb_enabled and global_step % self.log_every_n_steps == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        step_log = {f"train/step_{k}": v for k, v in loss_dict.items()}
                        step_log['learning_rate'] = current_lr
                        step_log['epoch'] = epoch + 1
                        wandb.log(step_log, step=global_step)

                # Log losses
                for key, value in loss_dict.items():
                    train_losses[key].append(value)

                # Update progress bar
                if batch_idx % self.log_every_n_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    postfix = {
                        'loss': f"{loss_dict['total_loss']:.4f}",
                        'lr': f"{current_lr:.2e}"
                    }
                    if 'focal_loss' in loss_dict:
                        postfix['focal'] = f"{loss_dict['focal_loss']:.4f}"
                    if 'ranking_loss' in loss_dict:
                        postfix['rank'] = f"{loss_dict['ranking_loss']:.4f}"
                    if 'mae_loss' in loss_dict:
                        postfix['mae'] = f"{loss_dict['mae_loss']:.4f}"
                    if 'band_loss' in loss_dict:
                        postfix['band'] = f"{loss_dict['band_loss']:.4f}"
                    if 'dist_loss' in loss_dict:
                        postfix['dist'] = f"{loss_dict['dist_loss']:.4f}"
                    pbar.set_postfix(postfix)

                # Periodic cache clearing
                if batch_idx % 50 == 0:
                    maybe_empty_cache()

            # Print epoch training summary
            print(f"\nEpoch {epoch + 1} Training Summary:")
            avg_train_losses = {}
            loss_keys = list(train_losses.keys())
            ordered_keys = []
            if 'total_loss' in loss_keys:
                ordered_keys.append('total_loss')
            ordered_keys.extend([k for k in loss_keys if k not in ordered_keys])
            for key in ordered_keys:
                avg_loss = np.mean(train_losses[key])
                avg_train_losses[key] = avg_loss
                print(f"  {key}: {avg_loss:.4f}")

            # IMPORTANT: Clear VRAM before validation to prevent OOM
            # (validation uses more chunks and no gradient checkpointing)
            torch.cuda.empty_cache()
            gc.collect()

            # Validation phase - MAE is PRIMARY metric
            val_metrics = self.validate()
            val_mae = val_metrics['mae']
            val_mse = val_metrics['mse']
            val_weighted_mse = val_metrics['weighted_mse']

            print(f"\nEpoch {epoch + 1} Validation Results:")
            print(f"  MAE: {val_mae:.4f} (PRIMARY METRIC)")
            print(f"  MSE: {val_mse:.4f}")
            print(f"  Weighted MSE: {val_weighted_mse:.4f}")

            # Wandb logging
            if self.wandb_enabled:
                current_lr = self.optimizer.param_groups[0]['lr']
                log_dict = {
                    'epoch': epoch + 1,
                    'val/mae': val_mae,
                    'val/mse': val_mse,
                    'val/weighted_mse': val_weighted_mse,
                    'val/mae_edge': val_metrics['mae_edge'],
                    'val/mae_mid': val_metrics['mae_mid'],
                    'val/pred_edge_pct': val_metrics['pred_edge_pct'],
                    'val/pred_mid_pct': val_metrics['pred_mid_pct'],
                    'val/true_edge_pct': val_metrics['true_edge_pct'],
                    'val/true_mid_pct': val_metrics['true_mid_pct'],
                    'learning_rate': current_lr,
                }
                for key, value in avg_train_losses.items():
                    log_dict[f"train/{key}"] = value
                wandb.log(log_dict, step=epoch + 1)

            # Save best model based on MAE
            if val_mae < self.best_val_mae:
                improvement = self.best_val_mae - val_mae
                self.best_val_mae = val_mae

                # Save state dict
                self.best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

                # Save to disk
                save_dir = self.config.checkpoint.save_dir
                os.makedirs(save_dir, exist_ok=True)

                checkpoint_path = os.path.join(
                    save_dir,
                    f"model_best_mae_{self.config.experiment_name}.pth"
                )

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_mae': val_mae,
                    'val_mse': val_mse,
                    'config': self.config.to_dict()
                }, checkpoint_path)

                print(f"\n✓ NEW BEST MAE: {val_mae:.4f} (improved by {improvement:.4f})")
                print(f"✓ Model saved to: {checkpoint_path}")

            elif val_mae > self.best_val_mae * 1.15:
                # Model degraded significantly (>15% worse)
                print(f"\n⚠ WARNING: Current MAE ({val_mae:.4f}) is >15% worse than best ({self.best_val_mae:.4f})")
                print("⚠ Reloading best model state...")

                if self.best_state_dict is not None:
                    self.model.load_state_dict(self.best_state_dict)
                    self.model.to(self.device)

            else:
                print(f"\nCurrent MAE: {val_mae:.4f} (Best: {self.best_val_mae:.4f})")

            # NEW: Save best acc|err|<=1.0 checkpoint
            if not hasattr(self, 'best_acc_err_le_1_0'):
                self.best_acc_err_le_1_0 = 0.0

            val_acc_err_le_1_0 = val_metrics['acc_err_le_1_0']
            if val_acc_err_le_1_0 > self.best_acc_err_le_1_0:
                improvement = val_acc_err_le_1_0 - self.best_acc_err_le_1_0
                self.best_acc_err_le_1_0 = val_acc_err_le_1_0

                # Ensure save_dir is defined
                save_dir = self.config.checkpoint.save_dir
                os.makedirs(save_dir, exist_ok=True)

                checkpoint_path_acc = os.path.join(
                    save_dir,
                    f"model_best_acc1_0_{self.config.experiment_name}.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_mae': val_mae,
                    'val_acc_err_le_1_0': val_acc_err_le_1_0,
                    'config': self.config.to_dict()
                }, checkpoint_path_acc)

                print(f"\n✓ NEW BEST ACC|err|<=1.0: {val_acc_err_le_1_0*100:.2f}% (improved by {improvement*100:.2f}pp)")
                print(f"✓ Checkpoint saved to: {checkpoint_path_acc}")
            else:
                print(f"Current ACC|err|<=1.0: {val_acc_err_le_1_0*100:.2f}% (Best: {self.best_acc_err_le_1_0*100:.2f}%)")

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best validation MAE: {self.best_val_mae:.4f}")
        print("=" * 80)

        # Load best model for final evaluation
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            self.model.to(self.device)

    def validate(self):
        """
        Validation loop

        Returns:
            Dictionary with comprehensive validation metrics:
            - mae: Mean Absolute Error (PRIMARY METRIC)
            - mse: Mean Squared Error
            - weighted_mse: Class-weighted MSE
            - mae_edge: MAE for edge scores
            - mae_mid: MAE for mid scores
            - pred_edge_pct: Percentage of edge predictions
            - pred_mid_pct: Percentage of mid predictions
            - true_edge_pct: Percentage of true edge scores
            - true_mid_pct: Percentage of true mid scores
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        all_weights = []
        all_logits = []  # NEW: Store logits for distribution metrics

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move to device
                audio = batch['audio'].to(self.device)
                true_scores = batch['score'].to(self.device)

                # STEP 3: Conditional input based on use_question_encoder
                if self.config.model.use_question_encoder:
                    question_input_ids = batch['question_input_ids'].to(self.device)
                    question_attention_mask = batch['question_attention_mask'].to(self.device)
                    response_input_ids = batch['response_input_ids'].to(self.device)
                    response_attention_mask = batch['response_attention_mask'].to(self.device)

                    # Forward pass
                    with autocast():
                        outputs = self.model(
                            question_input_ids=question_input_ids,
                            question_attention_mask=question_attention_mask,
                            response_input_ids=response_input_ids,
                            response_attention_mask=response_attention_mask,
                            audio=audio
                        )

                    pred_scores = outputs['expected_score']
                    logits = outputs['logits']  # NEW: Get logits for distribution metrics

                    # Compute weights
                    target_indexes = (true_scores * 2).long().clamp(0, 20)
                    weights = self.loss_weights[target_indexes]

                    # Store predictions
                    all_preds.append(pred_scores.cpu())
                    all_targets.append(true_scores.cpu())
                    all_weights.append(weights.cpu())
                    all_logits.append(logits.cpu())  # NEW: Store logits

                else:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    # Forward pass
                    with autocast():
                        outputs = self.model(input_ids, attention_mask, audio)

                pred_scores = outputs['expected_score']
                logits = outputs['logits']  # NEW: Get logits for distribution metrics

                # Compute weights
                target_indexes = (true_scores * 2).long().clamp(0, 20)
                weights = self.loss_weights[target_indexes]

                # Store predictions
                all_preds.append(pred_scores.cpu())
                all_targets.append(true_scores.cpu())
                all_weights.append(weights.cpu())
                all_logits.append(logits.cpu())  # NEW: Store logits

        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_weights = torch.cat(all_weights)
        all_logits = torch.cat(all_logits)  # NEW: Concatenate logits

        # IMPORTANT: Round predictions to nearest 0.5 (VSTEP scoring: 3.0, 3.5, 4.0, ..., 9.0)
        all_preds_rounded = torch.round(all_preds * 2) / 2
        all_preds_rounded = all_preds_rounded.clamp(0, 10)

        # Compute metrics
        # 1. MAE (PRIMARY METRIC) - using ROUNDED predictions
        mae = torch.abs(all_preds_rounded - all_targets).mean().item()

        # 2. MSE - using ROUNDED predictions
        mse = ((all_preds_rounded - all_targets) ** 2).mean().item()

        # 3. Weighted MSE - using ROUNDED predictions
        weighted_mse = (((all_preds_rounded - all_targets) ** 2) * all_weights).sum() / all_weights.sum()
        weighted_mse = weighted_mse.item()

        # 4. Per-range MAE (for monitoring mid-edge bias) - using ROUNDED predictions
        edge_threshold = self.config.training.edge_threshold
        edge_mask = (all_targets <= edge_threshold) | (all_targets >= (10 - edge_threshold))
        mid_mask = (all_targets > edge_threshold) & (all_targets < (10 - edge_threshold))

        mae_edge = torch.abs(all_preds_rounded[edge_mask] - all_targets[edge_mask]).mean().item() if edge_mask.sum() > 0 else 0.0
        mae_mid = torch.abs(all_preds_rounded[mid_mask] - all_targets[mid_mask]).mean().item() if mid_mask.sum() > 0 else 0.0

        # 5. Prediction distribution (detect bias) - using ROUNDED predictions
        pred_edge = ((all_preds_rounded <= edge_threshold) | (all_preds_rounded >= (10 - edge_threshold))).sum()
        pred_mid = ((all_preds_rounded > edge_threshold) & (all_preds_rounded < (10 - edge_threshold))).sum()

        pred_edge_pct = pred_edge.item() / len(all_preds) * 100
        pred_mid_pct = pred_mid.item() / len(all_preds) * 100
        true_edge_pct = edge_mask.sum().item() / len(all_targets) * 100
        true_mid_pct = mid_mask.sum().item() / len(all_targets) * 100

        # 6. Error distribution metrics (NEW - for %|err|<=1.0 target) - using ROUNDED predictions
        abs_errors = torch.abs(all_preds_rounded - all_targets)
        acc_err_le_0_5 = (abs_errors <= 0.5).float().mean().item()
        acc_err_le_1_0 = (abs_errors <= 1.0).float().mean().item()  # PRIMARY TARGET
        acc_err_le_1_5 = (abs_errors <= 1.5).float().mean().item()

        # 7. Distribution sharpness metrics (NEW - for KL/entropy loss monitoring)
        # NOTE: These use raw logits (not rounded), to monitor model confidence
        all_probs = F.softmax(all_logits, dim=-1)  # [N, 21]
        max_probs = all_probs.max(dim=-1)[0]  # [N]
        avg_sharpness = max_probs.mean().item()

        # Compute average entropy (lower = sharper predictions)
        log_probs = torch.log(all_probs + 1e-8)
        entropy = -(all_probs * log_probs).sum(dim=-1)  # [N]
        avg_entropy = entropy.mean().item()

        # Boundary clustering (count predictions near 3.5, 7.5) - using ROUNDED predictions
        near_3_5 = ((all_preds_rounded - 3.5).abs() < 0.25).float().mean().item()
        near_7_5 = ((all_preds_rounded - 7.5).abs() < 0.25).float().mean().item()

        # Log detailed metrics
        print(f"  Per-range MAE:")
        print(f"    Edge (≤{edge_threshold} or ≥{10-edge_threshold}): {mae_edge:.4f} (n={edge_mask.sum()})")
        print(f"    Mid ({edge_threshold}<score<{10-edge_threshold}): {mae_mid:.4f} (n={mid_mask.sum()})")
        print(f"  Prediction distribution:")
        print(f"    Edge predictions: {pred_edge_pct:.1f}% (true: {true_edge_pct:.1f}%)")
        print(f"    Mid predictions: {pred_mid_pct:.1f}% (true: {true_mid_pct:.1f}%)")
        print(f"  Error distribution:")
        print(f"    %|err|<=0.5: {acc_err_le_0_5*100:.2f}%")
        print(f"    %|err|<=1.0: {acc_err_le_1_0*100:.2f}%  ← TARGET: >90%")
        print(f"    %|err|<=1.5: {acc_err_le_1_5*100:.2f}%")
        print(f"  Distribution metrics:")
        print(f"    Avg sharpness (max prob): {avg_sharpness:.4f} (higher=better)")
        print(f"    Avg entropy: {avg_entropy:.4f} (lower=better)")
        print(f"    Predictions near 3.5: {near_3_5*100:.2f}%")
        print(f"    Predictions near 7.5: {near_7_5*100:.2f}%")

        return {
            'mae': mae,
            'mse': mse,
            'weighted_mse': weighted_mse,
            'mae_edge': mae_edge,
            'mae_mid': mae_mid,
            'pred_edge_pct': pred_edge_pct,
            'pred_mid_pct': pred_mid_pct,
            'true_edge_pct': true_edge_pct,
            'true_mid_pct': true_mid_pct,
            'acc_err_le_0_5': acc_err_le_0_5,
            'acc_err_le_1_0': acc_err_le_1_0,
            'acc_err_le_1_5': acc_err_le_1_5,
            'sharpness': avg_sharpness,
            'entropy': avg_entropy,
            'near_3_5_pct': near_3_5,
            'near_7_5_pct': near_7_5,
        }

    def test(self, log_large_errors=True, error_threshold=2.0):
        """
        Test evaluation with detailed metrics

        Args:
            log_large_errors: Whether to log samples with large errors
            error_threshold: Threshold for "large" error

        Returns:
            Dictionary with comprehensive metrics
        """
        print("\n" + "=" * 80)
        print("Running test evaluation...")
        print("=" * 80)

        self.model.eval()

        all_preds = []
        all_targets = []
        all_weights = []
        all_candidate_ids = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Move to device
                audio = batch['audio'].to(self.device)
                true_scores = batch['score'].to(self.device)

                # STEP 3: Conditional input based on use_question_encoder
                if self.config.model.use_question_encoder:
                    question_input_ids = batch['question_input_ids'].to(self.device)
                    question_attention_mask = batch['question_attention_mask'].to(self.device)
                    response_input_ids = batch['response_input_ids'].to(self.device)
                    response_attention_mask = batch['response_attention_mask'].to(self.device)

                    # Forward pass
                    with autocast():
                        outputs = self.model(
                            question_input_ids=question_input_ids,
                            question_attention_mask=question_attention_mask,
                            response_input_ids=response_input_ids,
                            response_attention_mask=response_attention_mask,
                            audio=audio
                        )
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    # Forward pass
                    with autocast():
                        outputs = self.model(input_ids, attention_mask, audio)

                pred_scores = outputs['expected_score']

                # Compute weights
                target_indexes = (true_scores * 2).long().clamp(0, 20)
                weights = self.loss_weights[target_indexes]

                # Store predictions
                all_preds.append(pred_scores.cpu())
                all_targets.append(true_scores.cpu())
                all_weights.append(weights.cpu())
                all_candidate_ids.extend(batch['candidate_id'])

        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_weights = torch.cat(all_weights)

        # IMPORTANT: Round predictions to nearest 0.5 (VSTEP scoring: 3.0, 3.5, 4.0, ..., 9.0)
        all_preds_rounded = torch.round(all_preds * 2) / 2
        all_preds_rounded = all_preds_rounded.clamp(0, 10)

        # Compute comprehensive metrics - using ROUNDED predictions
        errors = all_preds_rounded - all_targets
        abs_errors = torch.abs(errors)

        # Primary metrics
        mae = abs_errors.mean().item()
        mse = (errors ** 2).mean().item()
        rmse = np.sqrt(mse)

        # Weighted metrics
        weighted_mae = (abs_errors * all_weights).sum() / all_weights.sum()
        weighted_mse = ((errors ** 2) * all_weights).sum() / all_weights.sum()

        # Distribution metrics
        median_ae = abs_errors.median().item()
        std_ae = abs_errors.std().item()

        # Accuracy at different tolerances
        acc_0_5 = (abs_errors <= 0.5).float().mean().item()
        acc_1_0 = (abs_errors <= 1.0).float().mean().item()
        acc_1_5 = (abs_errors <= 1.5).float().mean().item()

        # Print results
        print("\n" + "=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        print(f"\nPrimary Metric:")
        print(f"  MAE (Mean Absolute Error):     {mae:.4f}")
        print(f"\nOther Metrics:")
        print(f"  MSE (Mean Squared Error):      {mse:.4f}")
        print(f"  RMSE (Root Mean Squared):      {rmse:.4f}")
        print(f"  Median Absolute Error:         {median_ae:.4f}")
        print(f"  Std of Absolute Error:         {std_ae:.4f}")
        print(f"\nWeighted Metrics:")
        print(f"  Weighted MAE:                  {weighted_mae:.4f}")
        print(f"  Weighted MSE:                  {weighted_mse:.4f}")
        print(f"\nAccuracy within tolerance:")
        print(f"  ±0.5 points: {acc_0_5 * 100:.2f}%")
        print(f"  ±1.0 points: {acc_1_0 * 100:.2f}%")
        print(f"  ±1.5 points: {acc_1_5 * 100:.2f}%")
        print("=" * 80)

        # Log large errors
        if log_large_errors:
            large_error_mask = abs_errors >= error_threshold
            num_large_errors = large_error_mask.sum().item()

            if num_large_errors > 0:
                print(f"\nSamples with |error| >= {error_threshold}:")
                print(f"Total: {num_large_errors} ({num_large_errors / len(abs_errors) * 100:.2f}%)")
                print("-" * 80)

                # Get indices of large errors
                large_error_indices = torch.where(large_error_mask)[0]

                # Sort by error magnitude (descending)
                sorted_indices = large_error_indices[
                    abs_errors[large_error_indices].argsort(descending=True)
                ]

                # Print top 20
                for idx in sorted_indices[:20]:
                    idx = idx.item()
                    cand_id = all_candidate_ids[idx]
                    pred = all_preds[idx].item()
                    target = all_targets[idx].item()
                    error = errors[idx].item()

                    print(f"  {cand_id}: Pred={pred:.2f}, True={target:.2f}, Error={error:+.2f}")

                print("-" * 80)

        # Prepare metrics dictionary
        metrics_dict = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'median_ae': median_ae,
            'std_ae': std_ae,
            'weighted_mae': weighted_mae.item(),
            'weighted_mse': weighted_mse.item(),
            'acc_0_5': acc_0_5,
            'acc_1_0': acc_1_0,
            'acc_1_5': acc_1_5,
            'num_large_errors': num_large_errors if log_large_errors else None
        }

        # Wandb logging for test metrics
        if self.wandb_enabled:
            wandb_test_dict = {
                'test/mae': mae,
                'test/mse': mse,
                'test/rmse': rmse,
                'test/median_ae': median_ae,
                'test/std_ae': std_ae,
                'test/weighted_mae': weighted_mae.item(),
                'test/weighted_mse': weighted_mse.item(),
                'test/acc_0.5': acc_0_5,
                'test/acc_1.0': acc_1_0,
                'test/acc_1.5': acc_1_5,
            }
            if log_large_errors:
                wandb_test_dict['test/num_large_errors'] = num_large_errors
                wandb_test_dict['test/large_error_rate'] = num_large_errors / len(abs_errors) * 100
            wandb.log(wandb_test_dict)

        return metrics_dict


if __name__ == "__main__":
    print("✓ Trainer module loaded successfully")
