"""
Inference script for OLD model (Wav2Vec2 + Qwen2 with 3-attention fusion)
Copy and adapted from train_W2VAudio_bycandidates_V2.py
Only performs inference, no training.
"""

import argparse
import json
import math
import os
import gc
import ast
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import yaml
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, Wav2Vec2Model, Wav2Vec2Processor
import librosa

# Import text processing from existing module
from text_processing import replace_repeats, is_low_content


# ----------------------
# Config Loader
# ----------------------
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


# ----------------------
# Audio Processing Functions
# ----------------------
def _process_audio_file(absolute_path, processor, sample_rate=16000, num_chunks=10, chunk_length_sec=30):
    """Process a single audio file."""
    audio, sr = librosa.load(absolute_path, sr=sample_rate)
    audio_chunks = fixed_chunk_audio(audio, sr, num_chunks=num_chunks, chunk_length_sec=chunk_length_sec)
    
    chunk_samples = int(chunk_length_sec * sample_rate)
    processed_chunks = []
    
    for chunk in audio_chunks:
        inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
        chunk_tensor = inputs.input_values.squeeze(0)
        
        if chunk_tensor.shape[0] < chunk_samples:
            pad_length = chunk_samples - chunk_tensor.shape[0]
            chunk_tensor = F.pad(chunk_tensor, (0, pad_length), 'constant', 0)
        elif chunk_tensor.shape[0] > chunk_samples:
            chunk_tensor = chunk_tensor[:chunk_samples]
            
        processed_chunks.append(chunk_tensor)
    
    audio_tensor = torch.stack(processed_chunks)
    del audio, audio_chunks
    gc.collect()
    return audio_tensor


def fixed_chunk_audio(audio, sr, num_chunks=10, chunk_length_sec=30):
    """Cuts audio into exactly num_chunks with each chunk of length chunk_length_sec."""
    chunk_samples = int(chunk_length_sec * sr)
    audio_length = len(audio)
    if audio_length < chunk_samples:
        audio = np.pad(audio, (0, chunk_samples - audio_length), mode='constant')
        audio_length = len(audio)
    
    if num_chunks == 1:
        starts = [0]
    else:
        max_start = audio_length - chunk_samples
        starts = np.linspace(0, max_start, num_chunks, dtype=int)
    
    chunks = []
    for start in starts:
        end = start + chunk_samples
        chunk = audio[start:end]
        chunks.append(chunk)
    return chunks


# ----------------------
# Dataset (copied from train_W2VAudio_bycandidates_V2.py)
# ----------------------
class ESLDatasetByCandidatesWithAudio(Dataset):
    def __init__(self, dataframe, criteria='grammar', audio_processor=None, num_chunks=10, chunk_length_sec=30):
        """
        Enhanced ESL Dataset that supports both text and audio.
        """
        self.criteria = criteria
        self.audio_processor = audio_processor
        self.num_chunks = num_chunks
        self.chunk_length_sec = chunk_length_sec

        self.candidate_ids = dataframe['Candidate_ID'].tolist()
        self.text_prefix = f"The following is a spoken English response by a non-native speaker. Grade the {criteria} score based on the transcript below:"
        self.question_type_map = {
            1: "Social Interaction: Answer sevaral questions about familiar topics",
            2: "Solution Discussion: Choose one option from a situation and justify your choice",
            3: "Topic Development: Present a given topic with supporting ideas and answer follow-up questions"
        }
        self.question_types = dataframe['question_type'].apply(ast.literal_eval).tolist()

        self.scores = dataframe[criteria].astype(float).tolist()
        raw_texts = dataframe['text'].apply(ast.literal_eval).tolist()
        self.texts = [[
            f"{self.text_prefix} [Question Type: {self.question_type_map.get(qtype, '')}] {t}"
            for t, qtype in zip(raw_text, question_types)
        ] for raw_text, question_types in zip(raw_texts, self.question_types)]
        
        # Audio paths
        self.absolute_paths = dataframe['absolute_paths'].apply(ast.literal_eval).tolist() if 'absolute_paths' in dataframe.columns else [None] * len(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            'Candidate_ID': self.candidate_ids[idx],
            'text': self.texts[idx],
            'score': torch.tensor(self.scores[idx], dtype=torch.float32),
            'question_type': self.question_types[idx],
            'absolute_path': self.absolute_paths[idx]
        }

        # Process audio if available
        if self.absolute_paths[idx] is not None and self.audio_processor is not None:
            item['audio'] = []
            item['has_audio'] = []
            for absolute_path in self.absolute_paths[idx]:
                try:
                    audio_tensor = _process_audio_file(
                        absolute_path, 
                        self.audio_processor,
                        num_chunks=self.num_chunks,
                        chunk_length_sec=self.chunk_length_sec
                    )
                    item['audio'].append(audio_tensor)
                    item['has_audio'].append(True)
                except Exception as e:
                    print(f"Error processing audio {absolute_path}: {e}")
                    chunk_samples = int(self.chunk_length_sec * 16000)
                    item['audio'].append(torch.zeros(self.num_chunks, chunk_samples))
                    item['has_audio'].append(False)
        else:
            chunk_samples = int(self.chunk_length_sec * 16000)
            item['audio'] = [torch.zeros(self.num_chunks, chunk_samples)]
            item['has_audio'] = [False]
            
        return item


def get_collate_fn_bycandidates_with_audio(tokenizer, max_length=2048):
    def collate_fn(batch):
        cand_texts = []
        cand_audios = []
        cand_IDs = []
        scores = []
        all_question_types = []
        
        for item in batch:
            # Text part
            cand_texts.append(" [SEP] ".join(item['text']))
            all_question_types.extend(item['question_type'])
            # Audio part
            chunks = [a if torch.is_tensor(a) else torch.tensor(a) for a in item['audio']]
            cand_audio = torch.cat(chunks, dim=0)
            cand_audios.append(cand_audio)
            # Label
            scores.append(item['score'])
            # Candidate ID
            cand_IDs.append(item['Candidate_ID'])

        encoded = tokenizer(
            cand_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Find max sizes in this batch
        max_chunks_in_batch = max([a.shape[0] for a in cand_audios if a is not None], default=1)
        max_waveform_in_batch = max([a.shape[1] for a in cand_audios if a is not None], default=1)

        padded_audios = []
        for a in cand_audios:
            if a is None:
                padded = torch.zeros((max_chunks_in_batch, max_waveform_in_batch), dtype=torch.float)
            else:
                C, L = a.shape
                pad_L = max_waveform_in_batch - L
                pad_C = max_chunks_in_batch - C
                padded = F.pad(a, (0, pad_L, 0, pad_C))
            padded_audios.append(padded)

        audio_tensor = torch.stack(padded_audios, dim=0)
        score_tensor = torch.stack(scores) if isinstance(scores[0], torch.Tensor) else torch.tensor(scores, dtype=torch.float)

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'question_type': torch.tensor(all_question_types, dtype=torch.long),
            'audio': audio_tensor,
            'score': score_tensor,
            'candidate_id': cand_IDs,
            "absolute_path": [item["absolute_path"] for item in batch]
        }

    return collate_fn


# ----------------------
# Model (copied from train_W2VAudio_bycandidates_V2.py)
# ----------------------
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, expected_seq_len=32, attn_proj=None, dropout=None):
        super().__init__()
        self.attn_proj = attn_proj or nn.Linear(hidden_dim, 1)
        init_scale = 1.0 / math.log(expected_seq_len)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        if dropout is not None and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, hidden_states, attention_mask=None, visualize=False):
        B, T, D = hidden_states.size()
        device = hidden_states.device

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.float32, device=device)

        raw_scores = self.attn_proj(hidden_states)
        scale_factor = self.scale * math.log(T)
        scaled_scores = raw_scores * scale_factor

        attn_mask = attention_mask.unsqueeze(-1)
        scaled_scores = scaled_scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = F.softmax(scaled_scores, dim=1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        pooled = torch.sum(attn_weights * hidden_states, dim=1)

        if visualize:
            return pooled, attn_weights
        else:
            return pooled


class ESLGradingModelByCandidatesWithAudio(nn.Module):
    def __init__(self, 
                 model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct', 
                 audio_encoder_id="jonatasgrosman/wav2vec2-large-xlsr-53-english",
                 pooling_dropout=0.3, 
                 regression_dropout=0.5, 
                 avg_last_k=4,
                 d_fuse=256):
        super().__init__()
        self.num_types = 3
        self.pooling_dropout = pooling_dropout
        self.regression_dropout = regression_dropout
        self.avg_last_k = avg_last_k
        self.d_fuse = d_fuse

        # ========== TEXT ENCODER ==========
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        if "qwen" in model_name.lower():
            self.encoder.config.use_cache = False

        text_hidden_size = self.encoder.config.hidden_size
        self.encoder.gradient_checkpointing_enable()

        # ========== AUDIO ENCODER ==========
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_id)
        self.audio_hidden_dim = self.audio_encoder.config.output_hidden_size
        
        # ========== PROJECTION LAYERS ==========
        self.audio_proj = nn.Linear(self.audio_hidden_dim, d_fuse)
        self.audio_norm = nn.LayerNorm(d_fuse)
        self.text_proj = nn.Linear(text_hidden_size, d_fuse)
        self.text_norm = nn.LayerNorm(d_fuse)
        
        # ========== 3 ATTENTION MECHANISMS ==========
        self.text_self_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.text_self_norm = nn.LayerNorm(d_fuse)
        
        self.text_to_audio_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.t2a_norm = nn.LayerNorm(d_fuse)
        
        self.audio_to_text_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.a2t_norm = nn.LayerNorm(d_fuse)
        
        # ========== 3 ATTENTION POOLING LAYERS ==========
        self.text_self_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, 256),
            nn.Tanh(), 
            nn.Dropout(pooling_dropout),
            nn.Linear(256, 1, bias=False)
        )
        self.text_self_pool = AttentionPooling(d_fuse, attn_proj=self.text_self_attn_proj, 
                                              expected_seq_len=512, dropout=pooling_dropout)
        
        self.t2a_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, 256),
            nn.Tanh(), 
            nn.Dropout(pooling_dropout),
            nn.Linear(256, 1, bias=False)
        )
        self.t2a_pool = AttentionPooling(d_fuse, attn_proj=self.t2a_attn_proj, 
                                        expected_seq_len=512, dropout=pooling_dropout)
        
        self.a2t_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, 256),
            nn.Tanh(), 
            nn.Dropout(pooling_dropout),
            nn.Linear(256, 1, bias=False)
        )
        self.a2t_pool = AttentionPooling(d_fuse, attn_proj=self.a2t_attn_proj, 
                                        expected_seq_len=10, dropout=pooling_dropout)
        
        # ========== REGRESSION HEAD ==========
        self.reg_head = nn.Sequential(
            nn.Linear(3 * d_fuse, 2 * d_fuse, bias=False),
            nn.LayerNorm(2 * d_fuse),
            nn.GELU(),
            nn.Dropout(regression_dropout),
            nn.Linear(2 * d_fuse, d_fuse, bias=False),
            nn.LayerNorm(d_fuse),
            nn.GELU(),
            nn.Dropout(regression_dropout),
            nn.Linear(d_fuse, 21, bias=False)
        )

    def encode_text(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states
        k = min(self.avg_last_k, len(all_hidden_states))
        if k == 1:
            hidden_states = all_hidden_states[-1]
        else:
            hidden_states = torch.stack(all_hidden_states[-k:], dim=0).mean(dim=0)
        hidden_states = hidden_states.float()
        return hidden_states

    def encode_audio(self, audio):
        if audio is None:
            return None
        
        batch_size, num_chunks, waveform_len = audio.shape
        device = next(self.parameters()).device
        
        audio_flat = audio.view(batch_size * num_chunks, waveform_len).to(device)
        
        with torch.no_grad():
            audio_features_flat = self.audio_encoder(input_values=audio_flat).last_hidden_state
            audio_features_flat = audio_features_flat.mean(dim=1)
        
        audio_features = audio_features_flat.view(batch_size, num_chunks, -1)
        audio_features = self.audio_proj(audio_features)
        audio_features = self.audio_norm(audio_features)
        
        return audio_features

    def apply_three_attention_mechanisms(self, text_features, audio_features, attention_mask):
        batch_size = text_features.size(0)
        device = text_features.device
        
        text_proj = self.text_proj(text_features)
        text_proj = self.text_norm(text_proj)
        
        # 1. Text Self-Attention
        text_self_output, _ = self.text_self_attention(
            query=text_proj, key=text_proj, value=text_proj
        )
        text_self_output = self.text_self_norm(text_self_output)
        
        with torch.amp.autocast('cuda', enabled=False):
            text_self_pooled = self.text_self_pool(text_self_output, attention_mask)
        
        if audio_features is None:
            t2a_pooled = torch.zeros(batch_size, self.d_fuse, device=device)
            a2t_pooled = torch.zeros(batch_size, self.d_fuse, device=device)
        else:
            # 2. Text-to-Audio Cross-Attention
            t2a_output, _ = self.text_to_audio_attention(
                query=text_proj, key=audio_features, value=audio_features
            )
            t2a_output = self.t2a_norm(t2a_output)
            
            with torch.amp.autocast('cuda', enabled=False):
                t2a_pooled = self.t2a_pool(t2a_output, attention_mask)
            
            # 3. Audio-to-Text Cross-Attention
            a2t_output, _ = self.audio_to_text_attention(
                query=audio_features, key=text_proj, value=text_proj
            )
            a2t_output = self.a2t_norm(a2t_output)
            
            with torch.amp.autocast('cuda', enabled=False):
                a2t_pooled = self.a2t_pool(a2t_output)
        
        return text_self_pooled, t2a_pooled, a2t_pooled

    def forward(self, input_ids, attention_mask, audio=None):
        text_hidden_states = self.encode_text(input_ids, attention_mask)
        audio_features = self.encode_audio(audio)
        
        text_self_pooled, t2a_pooled, a2t_pooled = self.apply_three_attention_mechanisms(
            text_hidden_states, audio_features, attention_mask
        )
        
        combined_features = torch.cat([text_self_pooled, t2a_pooled, a2t_pooled], dim=1)
        
        logits = self.reg_head(combined_features)
        probs = torch.softmax(logits, dim=-1)
        score_bins = torch.linspace(0, 10, steps=21).to(probs.device)
        expected_score = (probs * score_bins).sum(dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'expected_score': expected_score
        }

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'pooling_dropout': self.pooling_dropout,
                'regression_dropout': self.regression_dropout,
                'model_name': self.encoder.config._name_or_path,
                'avg_last_k': self.avg_last_k,
                'd_fuse': self.d_fuse
            }
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(
            model_name=config.get('model_name', 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'),
            pooling_dropout=config.get('pooling_dropout', 0.3),
            regression_dropout=config.get('regression_dropout', 0.5),
            avg_last_k=config.get('avg_last_k', 1),
            d_fuse=config.get('d_fuse', 256)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


# ----------------------
# Metrics
# ----------------------
def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    bins_true = np.round(y_true * 2) / 2.0
    bins_pred = np.round(y_pred * 2) / 2.0
    ints_true = np.round(bins_true * 2).astype(int)
    ints_pred = np.round(bins_pred * 2).astype(int)
    return cohen_kappa_score(ints_true, ints_pred, weights="quadratic")


# ----------------------
# Inference Functions
# ----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Inference for OLD ESL model (Wav2Vec2)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pth), overrides config")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save preds/metrics, overrides config")
    parser.add_argument("--limit", type=int, default=None, help="Optional: limit number of rows for quick test")
    return parser.parse_args()


def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.bos_token
    tok.padding_side = "right"
    return tok


def build_loader(csv_path: str, cfg, tokenizer, audio_processor, limit=None):
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)
    
    dataset = ESLDatasetByCandidatesWithAudio(
        df,
        criteria=cfg.data.criteria,
        audio_processor=audio_processor,
        num_chunks=cfg.audio.num_chunks,
        chunk_length_sec=cfg.audio.chunk_length_sec,
    )
    collate_fn = get_collate_fn_bycandidates_with_audio(
        tokenizer,
        max_length=cfg.data.max_length,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=cfg.inference.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return loader


def eval_split(name: str, loader: DataLoader, model, device: str, edge_thr: float):
    all_preds, all_true, all_ids = [], [], []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{name} evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio = batch["audio"].to(device)
            true_scores = batch["score"].to(device)

            with amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(input_ids, attention_mask, audio)

            pred_scores = outputs["expected_score"].detach().cpu().numpy()
            all_preds.append(pred_scores)
            all_true.append(true_scores.cpu().numpy())
            all_ids.extend(batch["candidate_id"])

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    # Round predictions to nearest 0.5 (VSTEP scoring)
    y_pred_rounded = np.round(y_pred * 2) / 2
    y_pred_rounded = np.clip(y_pred_rounded, 0, 10)

    # Compute metrics using ROUNDED predictions
    err = y_pred_rounded - y_true

    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = math.sqrt(mse)
    qwk = float(quadratic_weighted_kappa(y_true, y_pred_rounded))

    edge_mask = (y_true <= edge_thr) | (y_true >= (10 - edge_thr))
    mid_mask = ~edge_mask
    edge_mae = float(np.mean(np.abs(err[edge_mask]))) if edge_mask.any() else float("nan")
    mid_mae = float(np.mean(np.abs(err[mid_mask]))) if mid_mask.any() else float("nan")
    edge_pred_ratio = float(((y_pred_rounded <= edge_thr) | (y_pred_rounded >= (10 - edge_thr))).mean())
    edge_true_ratio = float(edge_mask.mean())

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "qwk": qwk,
        "edge_mae": edge_mae,
        "mid_mae": mid_mae,
        "edge_pred_ratio": edge_pred_ratio,
        "edge_true_ratio": edge_true_ratio,
        "n_edge": int(edge_mask.sum()),
        "n_mid": int(mid_mask.sum()),
        "n": int(len(y_true)),
    }

    print(f"\n{name} metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    df_out = pd.DataFrame({
        "Candidate_ID": all_ids,
        "true_score": y_true,
        "pred_score_raw": y_pred,
        "pred_score_rounded": y_pred_rounded,
        "error": err,
        "abs_error": np.abs(err),
    })
    return df_out, metrics


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get checkpoint path (args override config)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(cfg.checkpoint.load_checkpoint)
    if not ckpt_path or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Using checkpoint: {ckpt_path}")

    # Get output directory (args override config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from checkpoint...")
    model = ESLGradingModelByCandidatesWithAudio.load(str(ckpt_path), device=device)
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully")
    print(f"  Criteria: {cfg.data.criteria}")
    print(f"  d_fuse: {model.d_fuse}")

    # Build tokenizer and audio processor
    tokenizer = build_tokenizer(cfg.model.model_name)
    audio_processor = Wav2Vec2Processor.from_pretrained(cfg.model.audio_encoder_id)

    # Build data loader for test set
    print(f"\nLoading test data from: {cfg.data.test_path}")
    test_loader = build_loader(cfg.data.test_path, cfg, tokenizer, audio_processor, limit=args.limit)

    # Run evaluation
    edge_thr = cfg.inference.edge_threshold
    test_df, test_metrics = eval_split("TEST", test_loader, model, device, edge_thr)

    # Save results
    test_df.to_csv(output_dir / "test_predictions.csv", index=False)
    
    summary = {
        "test": test_metrics,
        "checkpoint": str(ckpt_path),
        "criteria": cfg.data.criteria,
        "edge_threshold": edge_thr,
        "config": cfg.to_dict(),
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}")
    print(f"  - test_predictions.csv")
    print(f"  - metrics_summary.json")

    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
