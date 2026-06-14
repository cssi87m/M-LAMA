"""
Configuration loading and management for ESL Speaking Grading Model
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing all configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return config_dict


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 4
    eval_batch_size: Optional[int] = None  # NEW: batch_size for eval (if None, use batch_size)
    accumulation_steps: int = 4
    epochs: int = 5
    base_lr: float = 2e-5
    encoder_lr: float = 2e-6
    scale_lr: float = 2e-4
    whisper_lr: float = 5e-6  # NEW: Learning rate for audio encoder (LoRA or full)
    warmup_steps: int = 50
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Loss weights
    lambda_mae: float = 0.8
    lambda_focal: float = 0.4
    lambda_ranking: float = 0.35
    lambda_dist: float = 0.25
    lambda_band: float = 0.2
    lambda_kl: float = 0.4  # NEW: KL divergence with one-hot targets
    lambda_entropy: float = 0.1  # NEW: Entropy penalty for flat distributions

    # Loss parameters
    focal_gamma: float = 2.0
    focal_beta: float = 1.0
    ranking_margin: float = 0.5
    edge_threshold: float = 3.5
    edge_penalty: float = 2.5
    mid_penalty: float = 2.5  # NEW: Penalty for mid→edge predictions
    band_margin: float = 1.0  # NEW: Margin for |err|<=1 band loss

    # KL/Entropy parameters
    use_hard_targets: bool = True  # True = one-hot targets, False = smoothed
    kl_temperature: float = 1.0  # Temperature for softening targets

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_name: str = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
    audio_encoder_id: str = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    audio_encoder_type: str = "wav2vec2"  # NEW: "wav2vec2" or "whisper"
    audio_encoder_frozen: bool = True  # NEW: Freeze encoder, train adapter only
    use_enhanced_hierarchical: bool = False  # NEW: Use PartAttentionPooling + chunk position embeddings
    d_fuse: int = 1024
    pooling_dropout: float = 0.3
    regression_dropout: float = 0.5
    avg_last_k: int = 4
    num_score_bins: int = 21
    num_parts: int = 3

    # Adapter config
    adapter_bottleneck_dim: int = 256
    adapter_num_heads: int = 8
    adapter_dropout: float = 0.1

    # Audio pooling strategy
    hierarchical_audio_pooling: bool = True  # If True: pool by parts [B,3*d_fuse]. If False: pool all chunks [B,d_fuse]

    # PHASE 2: Advanced architecture features
    use_gated_fusion: bool = False  # If True: use GatedMultimodalFusion, else: simple concatenation
    num_self_attn_layers: int = 1  # Number of self-attention layers (1=original, 2+=multi-layer with FFN)
    use_question_encoder: bool = False  # If True: encode questions separately with QuestionAwareEncoder

    # Text Encoder Configuration
    text_encoder_frozen: bool = False  # NEW: Freeze text encoder completely
    text_encoder_use_lora: bool = True
    text_encoder_lora_r: int = 32
    text_encoder_lora_alpha: int = 32
    text_encoder_lora_dropout: float = 0.1
    text_encoder_lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",     # Attention
        "gate_proj", "up_proj", "down_proj"         # FFN
    ])

    # Audio Encoder LoRA Configuration
    audio_encoder_use_lora: bool = True
    audio_encoder_lora_r: int = 16
    audio_encoder_lora_alpha: int = 16
    audio_encoder_lora_dropout: float = 0.05
    audio_encoder_lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "out_proj",   # Attention
        "fc1", "fc2"                                 # FFN
    ])

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    num_chunks: int = 10
    eval_num_chunks: Optional[int] = None  # NEW: num_chunks for eval (if None, use num_chunks)
    chunk_length_sec: int = 18  # 3 minutes / 10 chunks
    sample_rate: int = 16000
    max_audio_chunks: int = 30  # 3 parts × 10 chunks
    max_waveform_len: int = 288000  # 18s × 16kHz

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class DataConfig:
    """Data loading configuration"""
    train_path: str = "/home/user06/data/Speaking_VSTEP/Test_V2/train_V2_balance.csv"
    val_path: str = "/home/user06/data/Speaking_VSTEP/Test_V2/val_V2.csv"
    test_path: str = "/home/user06/data/Speaking_VSTEP/Test_V2/test_V2.csv"
    criteria: str = "final"  # Changed from 'grammar'
    max_length: int = 8192
    num_workers: int = 16
    prefetch_factor: int = 4

    # Sampling config
    sampling_alpha: float = 0.2
    class_weight_beta: float = 0.99
    edge_ratio: float = 0.3

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class CheckpointConfig:
    """Checkpoint loading/saving configuration"""
    load_checkpoint: Optional[str] = None
    save_dir: str = "./Model/checkpoints"
    save_best_only: bool = True
    monitor_metric: str = "val_mae"  # MAE is primary metric

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_dir: str = "./logs"
    wandb_project: str = "esl-audio-grading"
    wandb_enabled: bool = False
    log_every_n_steps: int = 10

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create from dictionary"""
        return cls(**config_dict)


@dataclass
class Config:
    """Complete configuration"""
    training: TrainingConfig
    model: ModelConfig
    audio: AudioConfig
    data: DataConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig
    experiment_name: str = "V2_final_score_1024dfuse_focal_ranking"

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml"):
        """
        Load complete configuration from YAML file

        Args:
            config_path: Path to YAML config file

        Returns:
            Config object with all settings
        """
        config_dict = load_config(config_path)

        return cls(
            training=TrainingConfig.from_dict(config_dict.get('training', {})),
            model=ModelConfig.from_dict(config_dict.get('model', {})),
            audio=AudioConfig.from_dict(config_dict.get('audio', {})),
            data=DataConfig.from_dict(config_dict.get('data', {})),
            checkpoint=CheckpointConfig.from_dict(config_dict.get('checkpoint', {})),
            logging=LoggingConfig.from_dict(config_dict.get('logging', {})),
            experiment_name=config_dict.get('experiment_name', "experiment")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'training': self.training.__dict__,
            'model': self.model.__dict__,
            'audio': self.audio.__dict__,
            'data': self.data.__dict__,
            'checkpoint': self.checkpoint.__dict__,
            'logging': self.logging.__dict__,
            'experiment_name': self.experiment_name
        }


if __name__ == "__main__":
    # Test loading config
    config = Config.from_yaml("config.yaml")
    print("✓ Configuration loaded successfully")
    print(f"Experiment: {config.experiment_name}")
    print(f"d_fuse: {config.model.d_fuse}")
    print(f"Primary metric: {config.checkpoint.monitor_metric}")
    print(f"Criteria: {config.data.criteria}")
