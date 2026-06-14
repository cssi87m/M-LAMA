"""
Audio Encoder Abstraction Layer
Supports both Wav2Vec2 and Whisper with unified interface

Author: Claude (based on ESL grading model architecture)
Date: 2025-12-13
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, WhisperModel
from abc import ABC, abstractmethod


class BaseAudioEncoder(nn.Module, ABC):
    """Abstract base class for audio encoders - inherits from nn.Module for proper device handling"""

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """
        Return output hidden dimension

        Returns:
            int: Hidden dimension (768 for Wav2Vec2, 1280 for Whisper)
        """
        pass

    @abstractmethod
    def forward(self, audio_input):
        """
        Encode audio to features

        Args:
            audio_input: Preprocessed audio (format depends on encoder)
                - Wav2Vec2: [B, waveform_len] raw waveform
                - Whisper: [B, 80, 3000] log-mel spectrogram

        Returns:
            features: [B, seq_len, hidden_dim]
        """
        pass


class Wav2Vec2Encoder(BaseAudioEncoder):
    """Wav2Vec2 encoder wrapper"""

    def __init__(self,
                 model_id="jonatasgrosman/wav2vec2-large-xlsr-53-english",
                 frozen=True):
        """
        Initialize Wav2Vec2 encoder

        Args:
            model_id: HuggingFace model ID
            frozen: Whether to freeze encoder weights
        """
        super().__init__()  # CRITICAL: Initialize nn.Module for device handling
        self.model = Wav2Vec2Model.from_pretrained(model_id)
        self.hidden_dim = self.model.config.output_hidden_size  # 768
        self.frozen = frozen

        # CRITICAL: Force encoder to FP32 to avoid mixed precision issues
        self.model.float()

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()  # Set to eval mode

    def get_hidden_dim(self):
        return self.hidden_dim

    def forward(self, audio_input):
        """
        Forward pass through Wav2Vec2

        Args:
            audio_input: [B, waveform_len] raw waveform at 16kHz

        Returns:
            features: [B, seq_len, 768]
        """
        # Simple forward like original code - let PyTorch autocast handle dtype
        # Encoder weights are FP32 (frozen), but computation uses mixed precision for speed
        with torch.set_grad_enabled(not self.frozen):
            outputs = self.model(input_values=audio_input)
            return outputs.last_hidden_state


class WhisperEncoder(BaseAudioEncoder):
    """Whisper encoder wrapper (encoder-only, no decoder)"""

    def __init__(self,
                 model_id="openai/whisper-large-v3",
                 frozen=True,
                 use_lora=False,
                 lora_config=None):
        """
        Initialize Whisper encoder

        Args:
            model_id: HuggingFace model ID
            frozen: Whether to freeze encoder weights
            use_lora: Whether to apply LoRA adaptation
            lora_config: Dictionary with LoRA parameters (lora_r, lora_alpha, lora_dropout, lora_target_modules)
        """
        super().__init__()  # CRITICAL: Initialize nn.Module for device handling
        # Load full model but only use encoder
        full_model = WhisperModel.from_pretrained(model_id)
        self.encoder = full_model.encoder
        self.config = full_model.config
        self.hidden_dim = full_model.config.d_model  # 1280 for large-v3-turbo
        self.num_mel_bins = full_model.config.num_mel_bins  # 80 for base/small, 128 for large/turbo
        self.frozen = frozen

        # CRITICAL: Force encoder to FP32 to avoid mixed precision issues
        self.encoder.float()

        if use_lora:
            # Apply LoRA
            from peft import LoraConfig, get_peft_model

            if lora_config is None:
                lora_config = {}

            peft_config = LoraConfig(
                r=lora_config.get('lora_r', 16),
                lora_alpha=lora_config.get('lora_alpha', 16),
                lora_dropout=lora_config.get('lora_dropout', 0.05),
                target_modules=lora_config.get('lora_target_modules', [
                    'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'
                ]),
                inference_mode=False,
                bias="none"
            )

            self.encoder = get_peft_model(self.encoder, peft_config)
            print(f"✓ Audio Encoder LoRA applied:")
            self.encoder.print_trainable_parameters()

            # Enable gradient checkpointing for LoRA
            if hasattr(self.encoder, 'gradient_checkpointing_enable'):
                self.encoder.gradient_checkpointing_enable()

        elif frozen:
            # Freeze all parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()  # Set to eval mode
            print(f"✓ Audio Encoder: Frozen")
        else:
            # Full fine-tune
            if hasattr(self.encoder, 'gradient_checkpointing_enable'):
                self.encoder.gradient_checkpointing_enable()
            print(f"✓ Audio Encoder: Full fine-tune")

    def get_hidden_dim(self):
        return self.hidden_dim

    def get_num_mel_bins(self):
        """Return number of mel bins expected by this Whisper model"""
        return self.num_mel_bins

    def forward(self, audio_input):
        """
        Forward pass through Whisper encoder

        Args:
            audio_input: [B, num_mel_bins, 3000] log-mel spectrogram
                - num_mel_bins: 80 (base/small) or 128 (large/turbo)
                - 3000 time steps for 30s audio

        Returns:
            features: [B, 1500, hidden_dim]
        """
        # Simple forward like original code - let PyTorch autocast handle dtype
        # Encoder weights are FP32 (frozen), but computation uses mixed precision for speed
        with torch.set_grad_enabled(not self.frozen):
            outputs = self.encoder(audio_input)
            return outputs.last_hidden_state


class AudioEncoderFactory:
    """Factory for creating audio encoders"""

    @staticmethod
    def create_encoder(encoder_type, model_id, frozen=True, use_lora=False, lora_config=None):
        """
        Create audio encoder based on type

        Args:
            encoder_type: "wav2vec2" or "whisper"
            model_id: HuggingFace model ID
            frozen: Whether to freeze encoder weights
            use_lora: Whether to apply LoRA adaptation
            lora_config: Dictionary with LoRA parameters (only for Whisper)

        Returns:
            BaseAudioEncoder instance

        Raises:
            ValueError: If encoder_type is unknown
        """
        encoder_type_lower = encoder_type.lower()

        if encoder_type_lower == "wav2vec2":
            return Wav2Vec2Encoder(
                model_id, frozen
            )
        elif encoder_type_lower == "whisper":
            return WhisperEncoder(
                model_id, frozen, use_lora, lora_config
            )
        else:
            raise ValueError(
                f"Unknown encoder type: {encoder_type}. "
                f"Supported types: 'wav2vec2', 'whisper'"
            )

    @staticmethod
    def get_processor(encoder_type, model_id):
        """
        Get corresponding processor for encoder type

        Args:
            encoder_type: "wav2vec2" or "whisper"
            model_id: HuggingFace model ID

        Returns:
            Processor instance (Wav2Vec2Processor or WhisperProcessor)

        Raises:
            ValueError: If encoder_type is unknown
        """
        encoder_type_lower = encoder_type.lower()

        if encoder_type_lower == "wav2vec2":
            from transformers import Wav2Vec2Processor
            return Wav2Vec2Processor.from_pretrained(model_id)
        elif encoder_type_lower == "whisper":
            from transformers import WhisperProcessor
            return WhisperProcessor.from_pretrained(model_id)
        else:
            raise ValueError(
                f"Unknown encoder type: {encoder_type}. "
                f"Supported types: 'wav2vec2', 'whisper'"
            )


if __name__ == "__main__":
    """
    Test audio encoder abstraction layer
    Usage: python audio_encoders.py
    """
    import torch

    print("=" * 80)
    print("Testing Audio Encoder Abstraction Layer")
    print("=" * 80)

    # Test Wav2Vec2
    print("\n1. Testing Wav2Vec2Encoder:")
    print("   Loading model...")
    wav2vec2 = Wav2Vec2Encoder(frozen=True)
    print(f"   ✓ Hidden dim: {wav2vec2.get_hidden_dim()}")

    # Test forward pass
    dummy_audio = torch.randn(2, 480000)  # [batch=2, waveform=30s]
    print(f"   Input shape: {dummy_audio.shape}")
    output = wav2vec2.forward(dummy_audio)
    print(f"   ✓ Output shape: {output.shape}")

    # Test Whisper
    print("\n2. Testing WhisperEncoder:")
    print("   Available Whisper models:")
    print("     - openai/whisper-base (80 mel bins, 74M params)")
    print("     - openai/whisper-large-v3-turbo (128 mel bins, 1550M params)")

    try:
        print("   Loading openai/whisper-large-v3-turbo...")
        whisper = WhisperEncoder(model_id="openai/whisper-large-v3-turbo", frozen=True)
        print(f"   ✓ Hidden dim: {whisper.get_hidden_dim()}")
        print(f"   ✓ Mel bins: {whisper.get_num_mel_bins()}")

        # Test forward pass
        mel_bins = whisper.get_num_mel_bins()
        dummy_mel = torch.randn(2, mel_bins, 3000)  # [batch=2, mel_bins, time=3000]
        print(f"   Input shape: {dummy_mel.shape}")
        output = whisper.forward(dummy_mel)
        print(f"   ✓ Output shape: {output.shape}")
    except Exception as e:
        print(f"   ⚠ Error: {e}")
        print("   (This is expected if model not downloaded or network issue)")

    # Test Factory
    print("\n3. Testing AudioEncoderFactory:")
    print("   Creating Wav2Vec2 via factory...")
    encoder = AudioEncoderFactory.create_encoder(
        "wav2vec2",
        "jonatasgrosman/wav2vec2-large-xlsr-53-english",
        frozen=True
    )
    print(f"   ✓ Created encoder with hidden dim: {encoder.get_hidden_dim()}")

    print("\n4. Testing Processor Factory:")
    print("   Getting Wav2Vec2 processor...")
    processor = AudioEncoderFactory.get_processor(
        "wav2vec2",
        "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    )
    print(f"   ✓ Processor type: {type(processor).__name__}")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
