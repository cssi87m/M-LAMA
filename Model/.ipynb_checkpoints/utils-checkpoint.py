"""
Utility functions for ESL Speaking Grading Model
Copied and adapted from train_W2VAudio_bycandidates_V2.py
"""

import torch
import numpy as np
import pandas as pd
import librosa
import gc
import asyncio
from typing import List
import sys
import os

# Add parent directory to path for text_processing import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model_old'))
from text_processing import replace_repeats, is_low_content


# ============================================================================
# Audio Processing Functions
# ============================================================================

async def preprocess_audio_wav2vec(absolute_path, processor, sample_rate=16000, num_chunks=10, chunk_length_sec=18):
    """
    Asynchronously preprocess audio file for the Wav2Vec2 model.
    Note: chunk_length_sec changed from 30 to 18 (3 minutes / 10 chunks)
    """
    try:
        loop = asyncio.get_event_loop()
        audio_tensor = await loop.run_in_executor(
            None,
            lambda: _process_audio_file(absolute_path, processor, sample_rate, num_chunks, chunk_length_sec)
        )
        return audio_tensor
    except Exception as e:
        print(f"Error in preprocessing audio: {str(e)}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


def _process_audio_file(absolute_path, processor, encoder_type='wav2vec2', sample_rate=16000, num_chunks=6, chunk_length_sec=30):
    """
    Process a single audio file for Wav2Vec2 or Whisper encoder.

    Args:
        absolute_path: Path to audio file
        processor: Wav2Vec2Processor or WhisperProcessor
        encoder_type: "wav2vec2" or "whisper"
        sample_rate: Audio sample rate (default 16000)
        num_chunks: Number of chunks (default 6)
        chunk_length_sec: Length of each chunk in seconds (default 30)

    Returns:
        Wav2Vec2: [num_chunks, chunk_samples] - raw waveform
        Whisper:  [num_chunks, num_mel_bins, 3000] - log-mel spectrogram
                  num_mel_bins = 80 (base/small) or 128 (large/turbo)
    """
    if not os.path.exists(absolute_path):
        print(f"WARNING: Audio path not found: {absolute_path}")
        raise FileNotFoundError(absolute_path)

    audio, sr = librosa.load(absolute_path, sr=sample_rate)
    audio_chunks = fixed_chunk_audio(audio, sr, num_chunks=num_chunks, chunk_length_sec=chunk_length_sec)

    processed_chunks = []

    if encoder_type.lower() == 'wav2vec2':
        # Wav2Vec2: raw waveform
        chunk_samples = int(chunk_length_sec * sample_rate)

        for chunk in audio_chunks:
            inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
            chunk_tensor = inputs.input_values.squeeze(0)  # [chunk_samples]

            # Pad/truncate to fixed length
            if chunk_tensor.shape[0] < chunk_samples:
                pad_length = chunk_samples - chunk_tensor.shape[0]
                chunk_tensor = torch.nn.functional.pad(chunk_tensor, (0, pad_length), 'constant', 0)
            elif chunk_tensor.shape[0] > chunk_samples:
                chunk_tensor = chunk_tensor[:chunk_samples]

            processed_chunks.append(chunk_tensor)

        audio_tensor = torch.stack(processed_chunks)  # [num_chunks, chunk_samples]

    elif encoder_type.lower() == 'whisper':
        # Whisper: log-mel spectrogram [num_mel_bins, 3000]
        # num_mel_bins: 80 (base/small) or 128 (large/turbo) - auto-determined by processor
        # Whisper expects 3000 frames for 30s audio
        target_time_steps = 3000  # Fixed: Whisper requirement for 30s chunks

        for chunk in audio_chunks:
            inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
            chunk_tensor = inputs.input_features.squeeze(0)  # [num_mel_bins, time_steps]

            # Pad/truncate time dimension (dim=1)
            current_time_steps = chunk_tensor.shape[1]
            if current_time_steps < target_time_steps:
                pad_length = target_time_steps - current_time_steps
                chunk_tensor = torch.nn.functional.pad(chunk_tensor, (0, pad_length), 'constant', 0)
            elif current_time_steps > target_time_steps:
                chunk_tensor = chunk_tensor[:, :target_time_steps]

            processed_chunks.append(chunk_tensor)

        audio_tensor = torch.stack(processed_chunks)  # [num_chunks, num_mel_bins, 3000]

    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Use 'wav2vec2' or 'whisper'.")

    del audio, audio_chunks
    gc.collect()
    return audio_tensor


def fixed_chunk_audio(audio, sr, num_chunks=6, chunk_length_sec=30):
    """
    Cuts audio into exactly num_chunks with each chunk of length chunk_length_sec.
    Default: 6 chunks × 30s = 3 minutes total

    Args:
        audio: Audio array
        sr: Sample rate
        num_chunks: Number of chunks (default 10)
        chunk_length_sec: Length of each chunk in seconds (default 18)

    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_length_sec * sr)
    audio_length = len(audio)

    # Pad if audio is too short
    if audio_length < chunk_samples:
        audio = np.pad(audio, (0, chunk_samples - audio_length), mode='constant')
        audio_length = len(audio)

    # Calculate start positions
    if num_chunks == 1:
        starts = [0]
    else:
        max_start = audio_length - chunk_samples
        starts = np.linspace(0, max_start, num_chunks, dtype=int)

    # Extract chunks
    chunks = []
    for start in starts:
        end = start + chunk_samples
        chunk = audio[start:end]
        chunks.append(chunk)

    return chunks


# ============================================================================
# Data Cleaning Functions
# ============================================================================

def clean_dataframe_bycandidates(df, remove_low_content=True, filter_scores=True, criteria='final'):
    """
    Cleans the dataframe by processing the 'text' field:
    - Applies replace_repeats
    - Optionally removes rows with low content using is_low_content

    Args:
        df: Input dataframe
        remove_low_content: Whether to remove low-content samples
        filter_scores: Whether to filter by scores
        criteria: Score column to use ('final' or 'grammar')
    """
    df = df.copy()
    df['text'] = df['text'].apply(lambda t: replace_repeats(t, k=2, tag="[REPEAT]"))

    if remove_low_content:
        mask = ~df['text'].apply(is_low_content)
        df = df[mask].reset_index(drop=True)

    if filter_scores:
        score_column = criteria if criteria in df.columns else 'final'
        mask = (df[score_column] >= 0.0)  # Keep all valid scores
        df = df[mask].reset_index(drop=True)
        print(f"After score filtering: {len(df)} samples")
        print(f"Score distribution:\n{df[score_column].value_counts().sort_index()}")

    return df


# ============================================================================
# Memory Management
# ============================================================================

def maybe_empty_cache(threshold=0.93):
    """
    Empty CUDA cache if memory usage exceeds threshold

    Args:
        threshold: Memory usage threshold (default 0.93 = 93%)
    """
    if torch.cuda.is_available():
        try:
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            if reserved / total > threshold:
                torch.cuda.empty_cache()
        except Exception:
            torch.cuda.empty_cache()


# ============================================================================
# Class Weighting Functions
# ============================================================================

def get_class_counts_from_dataframe(df, class_bins, criteria='final'):
    """
    Returns counts for each class bin (length = len(class_bins))

    Args:
        df: Input dataframe
        class_bins: List of score bins (e.g., [0.0, 0.5, 1.0, ..., 10.0])
        criteria: Score column to use ('final' or 'grammar')

    Returns:
        Array of counts for each bin
    """
    score_column = criteria if criteria in df.columns else 'final'
    class_to_index = {v: i for i, v in enumerate(class_bins)}
    indices = df[score_column].map(class_to_index)
    counts = np.zeros(len(class_bins), dtype=int)
    for idx in indices:
        if pd.notna(idx):  # Check for NaN
            counts[int(idx)] += 1
    return counts


def get_effective_number_weights(class_counts, beta=0.99):
    """
    Implements Cui et al. (2019) class-balanced loss weights

    Formula: w_i = (1 - beta) / (1 - beta^n_i)
    where n_i is the number of samples in class i

    Args:
        class_counts: Array of sample counts per class
        beta: Hyperparameter (default 0.99, was 0.9999 in old code)

    Returns:
        Tensor of normalized weights

    FIXED: Handle empty classes (class_counts = 0) to avoid NaN
    """
    # Fix: Clip class_counts to minimum 1 to avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    effective_num = 1.0 - np.power(beta, class_counts)
    # Fix: Add epsilon to denominator to avoid division by zero
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-7)
    weights = weights / np.mean(weights)  # normalize to mean 1
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# Optimizer Parameter Grouping
# ============================================================================

def get_param_groups(model, base_lr=1e-5, encoder_lr=1e-6, scale_lr=1e-3):
    """
    Group model parameters for differential learning rates

    Groups:
    - Special params (scale, alpha): scale_lr
    - Encoder params (text + audio encoders): encoder_lr
    - Other params (fusion, regression head): base_lr

    Args:
        model: PyTorch model
        base_lr: Learning rate for base parameters
        encoder_lr: Learning rate for encoder parameters
        scale_lr: Learning rate for scale/alpha parameters

    Returns:
        List of parameter groups for optimizer
    """
    special_params = []
    encoder_params = []
    base_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'scale' in name or 'alpha' in name:
            special_params.append(param)
        elif name.startswith('encoder.') or 'audio_encoder' in name:
            encoder_params.append(param)
        else:
            base_params.append(param)

    return [
        {'params': base_params, 'lr': base_lr},
        {'params': encoder_params, 'lr': encoder_lr},
        {'params': special_params, 'lr': scale_lr}
    ]


# ============================================================================
# Selective Freezing (Optional - not used in current plan)
# ============================================================================

def selective_freeze_embedding_layer(model, tokenizer, unfrozen_words):
    """
    Freezes the embedding layer of a transformer model,
    but allows selected tokens (from unfrozen_words) to remain trainable.

    Args:
        model: Hugging Face transformer model
        tokenizer: Corresponding tokenizer
        unfrozen_words: List or set of words to keep trainable
    """
    # Freeze the entire embedding layer
    embedding_layer = model.embeddings.word_embeddings
    embedding_layer.weight.requires_grad = True  # must stay True for masking
    for param in model.embeddings.parameters():
        param.requires_grad = True  # required for backward hook to work

    # Get token IDs of unfrozen words and all special tokens
    token_ids = set()
    for word in unfrozen_words:
        ids = tokenizer(word, add_special_tokens=False)['input_ids']
        token_ids.update(ids)

    # Add all special token IDs
    if hasattr(tokenizer, "all_special_ids"):
        token_ids.update(tokenizer.all_special_ids)
    else:
        # Fallback for tokenizers without all_special_ids
        for tok in tokenizer.all_special_tokens:
            ids = tokenizer(tok, add_special_tokens=False)['input_ids']
            token_ids.update(ids)

    vocab_size, hidden_size = embedding_layer.weight.shape
    grad_mask = torch.zeros(vocab_size, 1, device=embedding_layer.weight.device)
    for idx in token_ids:
        if idx < vocab_size:
            grad_mask[idx] = 1.0

    # Register gradient hook to zero out updates for frozen tokens
    def hook_fn(grad):
        return grad * grad_mask

    embedding_layer.weight.register_hook(hook_fn)


if __name__ == "__main__":
    # Test functions
    print("✓ Utils module loaded successfully")

    # Test class weight calculation
    class_counts = np.array([100, 200, 500, 300, 100])
    weights = get_effective_number_weights(class_counts, beta=0.99)
    print(f"Sample weights: {weights}")
    print(f"Mean weight: {weights.mean():.3f}")
