"""
ESL Speaking Grading Model with Audio

Key components:
- AudioBottleneckAdapter: Trainable adapter for frozen Wav2Vec2
- ESLGradingModelByCandidatesWithAudio: Main model with text + audio fusion
- Improvements: d_fuse=1024, part embeddings, learnable importance weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel, AutoConfig, Wav2Vec2Model


# ============================================================================
# Audio Bottleneck Adapter
# ============================================================================

class AudioBottleneckAdapter(nn.Module):
    """
    Bottleneck Adapter with Self-Attention
    Inspired by Houlsby et al. (2019) - Parameter-Efficient Transfer Learning

    Architecture: input (768) → down_proj (256) → Self-Attention → FFN → up_proj (768) → residual
    """

    def __init__(self, input_dim=768, bottleneck_dim=256, num_heads=8, dropout=0.1):
        super().__init__()

        # Bottleneck projection
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.layer_norm1 = nn.LayerNorm(bottleneck_dim)

        # Self-attention in bottleneck space
        self.self_attn = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(bottleneck_dim)

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim)
        )
        self.layer_norm3 = nn.LayerNorm(bottleneck_dim)

        # Up projection
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

        # Residual scaling (learnable)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        """
        Args:
            x: [B, seq_len, 768] or [B, 768]
        Returns:
            output: [B, seq_len, 768] or [B, 768]
        """
        residual = x

        # Down projection
        h = self.down_proj(x)  # [B, ?, 256]
        h = self.layer_norm1(h)

        # Self-attention (if sequence dimension exists)
        if h.dim() == 3:
            attn_out, _ = self.self_attn(h, h, h)
            h = h + attn_out
            h = self.layer_norm2(h)

        # Feedforward
        ffn_out = self.ffn(h)
        h = h + ffn_out
        h = self.layer_norm3(h)

        # Up projection
        output = self.up_proj(h)  # [B, ?, 768]

        # Residual connection with learnable scale
        output = residual + self.residual_scale * output

        return output


# ============================================================================
# Attention Pooling
# ============================================================================

class AttentionPooling(nn.Module):
    """Attention-based pooling with learnable scale"""

    def __init__(self, hidden_dim, expected_seq_len=32, attn_proj=None, dropout=None):
        super().__init__()
        self.attn_proj = attn_proj or nn.Linear(hidden_dim, 1)
        init_scale = 1.0 / math.log(expected_seq_len)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

        if dropout is not None and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [B, T, D]
            attention_mask: [B, T] (1 = keep, 0 = pad); optional
        Returns:
            pooled: [B, D]
        """
        B, T, D = hidden_states.size()
        device = hidden_states.device

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.float32, device=device)

        raw_scores = self.attn_proj(hidden_states)  # [B, T, 1]
        scale_factor = self.scale * math.log(T)
        scaled_scores = raw_scores * scale_factor  # [B, T, 1]

        attn_mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
        # Use dtype-safe mask value to avoid overflow under autocast/float16
        mask_value = torch.finfo(scaled_scores.dtype).min
        scaled_scores = scaled_scores.masked_fill(attn_mask == 0, mask_value)

        attn_weights = F.softmax(scaled_scores, dim=1)  # [B, T, 1]

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        pooled = torch.sum(attn_weights * hidden_states, dim=1)  # [B, D]

        return pooled


# ============================================================================
# Part Attention Pooling (NEW: Enhanced Hierarchical)
# ============================================================================

class PartAttentionPooling(nn.Module):
    """
    Attention pooling for chunks within a part
    Replaces simple mean pooling with learned attention weights

    This module learns which chunks are more important within each part
    (e.g., fluent segments vs disfluent segments)
    """

    def __init__(self, d_fuse=512, num_heads=4):
        """
        Initialize Part Attention Pooling

        Args:
            d_fuse: Feature dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_fuse,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_fuse)
        self.query = nn.Parameter(torch.randn(1, 1, d_fuse))  # Learnable query

    def forward(self, chunks):
        """
        Pool chunks using learned attention

        Args:
            chunks: [B, chunks_per_part, d_fuse] (e.g., [B, 6, 512])

        Returns:
            pooled: [B, d_fuse]
        """
        B = chunks.size(0)
        query = self.query.expand(B, -1, -1)  # [B, 1, d_fuse]

        # Attention pooling: query attends to all chunks
        attn_out, _ = self.attn(query, chunks, chunks)  # [B, 1, d_fuse]

        # Normalize and squeeze
        pooled = self.norm(attn_out.squeeze(1))  # [B, d_fuse]

        return pooled


class QuestionAwareEncoder(nn.Module):
    """
    Question-Aware Encoder sử dụng Cross-Attention (STEP 3)

    Kiến trúc:
        1. Question self-attention (hiểu tiêu chí chấm điểm)
        2. Cross-attention: Response (Q) attend vào Question (K, V)
        3. Pooling riêng cho question và response
        4. Gated fusion của question và response features

    Input:
        - question_features: [B, Q_len, d_fuse] - Embeddings token câu hỏi
        - response_features: [B, R_len, d_fuse] - Embeddings token câu trả lời
        - question_mask: [B, Q_len] - Attention mask câu hỏi
        - response_mask: [B, R_len] - Attention mask câu trả lời

    Output:
        - question_aware_features: [B, d_fuse] - Features pooled nhận biết câu hỏi
    """

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Question self-attention (hiểu tiêu chí chấm điểm)
        self.question_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.question_self_norm = nn.LayerNorm(d_model)

        # Cross-attention: Response attend vào Question
        self.question_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)

        # Pooling riêng cho questions và responses
        self.question_pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1, bias=False)
        )
        self.question_pool = AttentionPooling(
            d_model,
            attn_proj=self.question_pool_proj,
            expected_seq_len=128,  # Câu hỏi ngắn hơn
            dropout=dropout
        )

        self.response_pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1, bias=False)
        )
        self.response_pool = AttentionPooling(
            d_model,
            attn_proj=self.response_pool_proj,
            expected_seq_len=512,  # Câu trả lời dài hơn
            dropout=dropout
        )

        # Gated fusion của question và response
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, question_features, response_features,
                question_mask=None, response_mask=None):
        """
        Forward pass với cross-attention

        Returns:
            question_aware_features: [B, d_model]
        """
        # 1. Question self-attention (hiểu tiêu chí chấm điểm)
        q_self_out, _ = self.question_self_attn(
            question_features, question_features, question_features,
            key_padding_mask=(question_mask == 0) if question_mask is not None else None
        )
        question_features = self.question_self_norm(question_features + q_self_out)

        # 2. Cross-attention: Response attend vào Question
        r_cross_out, cross_attn_weights = self.question_cross_attn(
            query=response_features,
            key=question_features,
            value=question_features,
            key_padding_mask=(question_mask == 0) if question_mask is not None else None
        )
        response_aware = self.cross_attn_norm(response_features + r_cross_out)

        # 3. Pool cả hai sequences
        question_pooled = self.question_pool(question_features, question_mask)  # [B, d_model]
        response_pooled = self.response_pool(response_aware, response_mask)     # [B, d_model]

        # 4. Gated fusion
        combined = torch.cat([question_pooled, response_pooled], dim=-1)  # [B, 2*d_model]
        gate_weight = self.gate(combined)  # [B, d_model], giá trị trong [0, 1]

        question_aware_features = gate_weight * question_pooled + (1 - gate_weight) * response_pooled

        return question_aware_features


# ============================================================================
# Transformer Building Blocks (PHASE 2)
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Standard Transformer block: Self-Attn + FFN + Residual + LayerNorm
    Adds FFN layers that were missing in original architecture
    """
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN (CRITICAL - missing in current code!)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class MultiLayerSelfAttention(nn.Module):
    """
    Multi-layer self-attention (2-3 layers)
    Replaces single-layer self-attention for deeper representations
    """
    def __init__(self, d_fuse=512, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_fuse, num_heads, d_fuse*4, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)  # Self-attn + FFN + residual
        return x


# ============================================================================
# Gated Multimodal Fusion (PHASE 2)
# ============================================================================

class GatedMultimodalFusion(nn.Module):
    """
    Gated fusion with learned modality interactions
    Replaces simple concatenation with learned fusion
    Inspired by MFN (Memory Fusion Network) and MISA
    """
    def __init__(self, d_fuse=512, num_modalities=5, dropout=0.3):
        super().__init__()
        self.d_fuse = d_fuse
        self.num_modalities = num_modalities

        # Modality-specific transformations
        self.text_self_transform = nn.Linear(d_fuse, d_fuse)
        self.audio_self_transform = nn.Linear(d_fuse, d_fuse)
        self.t2a_transform = nn.Linear(d_fuse, d_fuse)
        self.a2t_transform = nn.Linear(d_fuse, d_fuse)
        self.audio_mean_transform = nn.Linear(d_fuse, d_fuse)

        # Gating network (learn importance of each modality)
        self.gate_net = nn.Sequential(
            nn.Linear(num_modalities * d_fuse, d_fuse * 2),
            nn.LayerNorm(d_fuse * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fuse * 2, num_modalities),
            nn.Softmax(dim=-1)  # [B, 5] weights
        )

        # Cross-modality interaction (low-rank bilinear)
        self.text_audio_interaction = nn.Bilinear(d_fuse, d_fuse, d_fuse)
        self.cross_attn_interaction = nn.Bilinear(d_fuse, d_fuse, d_fuse)

        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_fuse * 3, d_fuse * 2),  # 3 = weighted_sum + 2 interactions
            nn.LayerNorm(d_fuse * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fuse * 2, d_fuse),
            nn.LayerNorm(d_fuse),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, text_self, audio_self, t2a, a2t, audio_mean):
        """
        Args:
            All inputs: [B, d_fuse]

        Returns:
            fused: [B, d_fuse] - rich multimodal representation
        """
        B = text_self.size(0)

        # Transform each modality
        text_self_t = self.text_self_transform(text_self)
        audio_self_t = self.audio_self_transform(audio_self)
        t2a_t = self.t2a_transform(t2a)
        a2t_t = self.a2t_transform(a2t)
        audio_mean_t = self.audio_mean_transform(audio_mean)

        # Stack modalities: [B, 5, d_fuse]
        modalities = torch.stack([
            text_self_t, audio_self_t, t2a_t, a2t_t, audio_mean_t
        ], dim=1)

        # Learn modality importance weights
        concat_flat = modalities.view(B, -1)  # [B, 5*d_fuse]
        weights = self.gate_net(concat_flat)  # [B, 5]

        # Weighted sum: [B, d_fuse]
        weighted_sum = (modalities * weights.unsqueeze(-1)).sum(1)

        # Cross-modality interactions (bilinear)
        text_audio_inter = self.text_audio_interaction(text_self_t, audio_self_t)
        cross_attn_inter = self.cross_attn_interaction(t2a_t, a2t_t)

        # Concatenate: weighted_sum + 2 interactions
        combined = torch.cat([
            weighted_sum,       # [B, d_fuse]
            text_audio_inter,   # [B, d_fuse]
            cross_attn_inter    # [B, d_fuse]
        ], dim=-1)  # [B, 3*d_fuse]

        # Final fusion
        fused = self.fusion_mlp(combined)  # [B, d_fuse]

        return fused


# ============================================================================
# Main Model
# ============================================================================

class ESLGradingModelByCandidatesWithAudio(nn.Module):
    """
    ESL Speaking Grading Model with Text + Audio Fusion

    Improvements from v1:
    - d_fuse: 256 → 1024 for better capacity
    - Audio Adapter: Trainable bottleneck adapter on frozen Wav2Vec2
    - Part embeddings: Learnable position embeddings for 3 parts
    - Part importance: Learnable weights for task importance
    """

    def __init__(self,
                 model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct',
                 audio_encoder_type='wav2vec2',  # NEW: "wav2vec2" or "whisper"
                 audio_encoder_id="jonatasgrosman/wav2vec2-large-xlsr-53-english",
                 audio_encoder_frozen=True,  # NEW: freeze encoder, train adapter only
                 pooling_dropout=0.3,
                 regression_dropout=0.5,
                 avg_last_k=4,
                 d_fuse=1024,  # Increased from 256
                 adapter_bottleneck_dim=256,
                 adapter_num_heads=8,
                 adapter_dropout=0.1,
                 num_parts=3,
                 hierarchical_audio_pooling=True,  # Hierarchical pooling config
                 use_enhanced_hierarchical=False,  # NEW: Enhanced with attention pooling
                 use_gated_fusion=False,  # PHASE 2: Use GatedMultimodalFusion
                 num_self_attn_layers=1,  # PHASE 2: Number of self-attention layers (1=original, 2+=deeper)
                 use_question_encoder=False,  # PHASE 2: Use QuestionAwareEncoder (requires dataloader changes)
                 # Text Encoder parameters
                 text_encoder_frozen=False,  # NEW: Freeze text encoder completely
                 text_encoder_use_lora=True,
                 text_encoder_lora_r=32,
                 text_encoder_lora_alpha=32,
                 text_encoder_lora_dropout=0.1,
                 text_encoder_lora_target_modules=None,
                 # Audio Encoder LoRA parameters
                 audio_encoder_use_lora=True,
                 audio_encoder_lora_r=16,
                 audio_encoder_lora_alpha=16,
                 audio_encoder_lora_dropout=0.05,
                 audio_encoder_lora_target_modules=None):
        super().__init__()

        self.pooling_dropout = pooling_dropout
        self.regression_dropout = regression_dropout
        self.avg_last_k = avg_last_k
        self.d_fuse = d_fuse
        self.num_parts = num_parts
        self.hierarchical_audio_pooling = hierarchical_audio_pooling
        self.use_enhanced_hierarchical = use_enhanced_hierarchical  # NEW
        self.audio_encoder_type = audio_encoder_type  # NEW
        self.audio_encoder_frozen = audio_encoder_frozen
        self.audio_encoder_use_lora = audio_encoder_use_lora
        self.text_encoder_frozen = text_encoder_frozen  # NEW
        self.text_encoder_use_lora = text_encoder_use_lora
        self.use_gated_fusion = use_gated_fusion  # PHASE 2
        self.num_self_attn_layers = num_self_attn_layers  # PHASE 2
        self.use_question_encoder = use_question_encoder  # PHASE 2

        # ========== TEXT ENCODER ==========
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)

        if "qwen" in model_name.lower():
            self.encoder.config.use_cache = False

        text_hidden_size = self.encoder.config.hidden_size  # 1536 for Qwen2-1.5B

        # Text encoder training mode selection
        if text_encoder_frozen:
            # FROZEN: Freeze all parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"✓ Text Encoder: FROZEN (0 trainable parameters)")
        elif text_encoder_use_lora:
            from peft import LoraConfig, get_peft_model, TaskType

            # Set default target modules if not provided
            if text_encoder_lora_target_modules is None:
                text_encoder_lora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",     # Attention
                    "gate_proj", "up_proj", "down_proj"         # FFN
                ]

            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=text_encoder_lora_r,
                lora_alpha=text_encoder_lora_alpha,
                lora_dropout=text_encoder_lora_dropout,
                target_modules=text_encoder_lora_target_modules,
                inference_mode=False,
                bias="none"
            )

            self.encoder = get_peft_model(self.encoder, lora_config)
            print(f"✓ Text Encoder LoRA applied:")
            self.encoder.print_trainable_parameters()
        else:
            # Full fine-tune mode
            self.encoder.gradient_checkpointing_enable()
            print(f"✓ Text Encoder: Full fine-tune mode")

        # ========== AUDIO ENCODER (NEW: Abstracted) ==========
        from audio_encoders import AudioEncoderFactory

        # Prepare LoRA config dict for audio encoder
        if audio_encoder_lora_target_modules is None:
            audio_encoder_lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "out_proj",   # Attention
                "fc1", "fc2"                                 # FFN
            ]

        audio_lora_config = {
            'lora_r': audio_encoder_lora_r,
            'lora_alpha': audio_encoder_lora_alpha,
            'lora_dropout': audio_encoder_lora_dropout,
            'lora_target_modules': audio_encoder_lora_target_modules
        } if audio_encoder_use_lora else None

        self.audio_encoder = AudioEncoderFactory.create_encoder(
            encoder_type=audio_encoder_type,
            model_id=audio_encoder_id,
            frozen=audio_encoder_frozen,
            use_lora=audio_encoder_use_lora,
            lora_config=audio_lora_config
        )
        self.audio_hidden_dim = self.audio_encoder.get_hidden_dim()  # 768 for Wav2Vec2, 1280 for Whisper

        # IMPROVEMENT: Audio Bottleneck Adapter (trainable, dimension-adaptive)
        self.audio_adapter = AudioBottleneckAdapter(
            input_dim=self.audio_hidden_dim,  # Adapts to 768 or 1280
            bottleneck_dim=adapter_bottleneck_dim,
            num_heads=adapter_num_heads,
            dropout=adapter_dropout
        )

        # ========== PROJECTION LAYERS ==========
        # Audio projection to common space
        self.audio_proj = nn.Linear(self.audio_hidden_dim, d_fuse)
        self.audio_norm = nn.LayerNorm(d_fuse)

        # Text projection to common space
        self.text_proj = nn.Linear(text_hidden_size, d_fuse)
        self.text_norm = nn.LayerNorm(d_fuse)

        # IMPROVEMENT: Part position embeddings
        self.part_position_embeds = nn.Parameter(torch.randn(num_parts, text_hidden_size))
        self.part_embed_proj = nn.Linear(text_hidden_size, d_fuse)  # Project to fusion space

        # IMPROVEMENT: Learnable part importance weights
        self.part_importance = nn.Parameter(torch.ones(num_parts))

        # NEW: Enhanced Hierarchical Pooling components
        if use_enhanced_hierarchical and hierarchical_audio_pooling:
            self.part_attn_pooling = PartAttentionPooling(d_fuse, num_heads=4)
            self.chunk_position_embeds = nn.Parameter(torch.randn(6, d_fuse))  # 6 chunks per part

        # STEP 3: Question-Aware Encoder (optional)
        if use_question_encoder:
            self.question_aware_encoder = QuestionAwareEncoder(
                d_model=d_fuse,  # 1024
                num_heads=8,
                dropout=pooling_dropout
            )
            print("✓ Question-Aware Encoder enabled")

        # ========== 4 ATTENTION MECHANISMS ==========
        # PHASE 2: Use multi-layer self-attention if num_self_attn_layers > 1

        # 1. Text Self-Attention
        if num_self_attn_layers > 1:
            # PHASE 2: Multi-layer self-attention with FFN
            self.text_self_attention = MultiLayerSelfAttention(
                d_fuse=d_fuse,
                num_layers=num_self_attn_layers,
                num_heads=8,
                dropout=pooling_dropout
            )
            self.text_self_norm = None  # Not needed, included in MultiLayerSelfAttention
        else:
            # Original single-layer
            self.text_self_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
            self.text_self_norm = nn.LayerNorm(d_fuse)

        # 2. Text-to-Audio Cross-Attention
        self.text_to_audio_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.t2a_norm = nn.LayerNorm(d_fuse)

        # 3. Audio-to-Text Cross-Attention
        self.audio_to_text_attention = nn.MultiheadAttention(embed_dim=d_fuse, num_heads=8, batch_first=True)
        self.a2t_norm = nn.LayerNorm(d_fuse)

        # 4. Audio Self-Attention (similar to Text Self-Attention)
        if num_self_attn_layers > 1:
            # PHASE 2: Multi-layer self-attention with FFN
            self.audio_self_attention = MultiLayerSelfAttention(
                d_fuse=d_fuse,
                num_layers=num_self_attn_layers,
                num_heads=8,
                dropout=pooling_dropout
            )
            self.audio_self_norm = None  # Not needed, included in MultiLayerSelfAttention
        else:
            # Original single-layer
            self.audio_self_attention = nn.MultiheadAttention(
                embed_dim=d_fuse,
                num_heads=8,
                dropout=pooling_dropout,
                batch_first=True
            )
            self.audio_self_norm = nn.LayerNorm(d_fuse)

        # ========== 3 ATTENTION POOLING LAYERS ==========
        # Text self-attention pooling
        self.text_self_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, d_fuse // 4),
            nn.Tanh(),
            nn.Dropout(pooling_dropout),
            nn.Linear(d_fuse // 4, 1, bias=False)
        )
        self.text_self_pool = AttentionPooling(
            d_fuse,
            attn_proj=self.text_self_attn_proj,
            expected_seq_len=512,
            dropout=pooling_dropout
        )

        # Text-to-audio pooling
        self.t2a_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, d_fuse // 4),
            nn.Tanh(),
            nn.Dropout(pooling_dropout),
            nn.Linear(d_fuse // 4, 1, bias=False)
        )
        self.t2a_pool = AttentionPooling(
            d_fuse,
            attn_proj=self.t2a_attn_proj,
            expected_seq_len=512,
            dropout=pooling_dropout
        )

        # Audio-to-text pooling
        self.a2t_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, d_fuse // 4),
            nn.Tanh(),
            nn.Dropout(pooling_dropout),
            nn.Linear(d_fuse // 4, 1, bias=False)
        )
        self.a2t_pool = AttentionPooling(
            d_fuse,
            attn_proj=self.a2t_attn_proj,
            expected_seq_len=18,  # matches max_audio_chunks (3 parts × 6 chunks)
            dropout=pooling_dropout
        )

        # NEW: Audio self-attention pooling
        self.audio_self_attn_proj = nn.Sequential(
            nn.Linear(d_fuse, d_fuse // 4),
            nn.Tanh(),
            nn.Dropout(pooling_dropout),
            nn.Linear(d_fuse // 4, 1, bias=False)
        )
        self.audio_self_pool = AttentionPooling(
            d_fuse,
            attn_proj=self.audio_self_attn_proj,
            expected_seq_len=num_parts,  # 3 parts
            dropout=pooling_dropout
        )

        # ========== GATED MULTIMODAL FUSION (PHASE 2) ==========
        if use_gated_fusion:
            self.gated_fusion = GatedMultimodalFusion(
                d_fuse=d_fuse,
                num_modalities=5,
                dropout=pooling_dropout
            )

        # ========== REGRESSION HEAD ==========
        # PHASE 2: Deeper 5-layer regression head
        # Input: d_fuse from GatedMultimodalFusion (or 5*d_fuse from concatenation if gated fusion disabled)
        # For backward compatibility, we detect the input size dynamically
        # When use_gated_fusion=True: input is d_fuse
        # When use_gated_fusion=False: input is 5*d_fuse (fallback to concatenation)
        self.reg_head = nn.Sequential(
            # Layer 1: → d_fuse*3
            nn.Linear(d_fuse, d_fuse * 3, bias=False),
            nn.LayerNorm(d_fuse * 3),
            nn.GELU(),
            nn.Dropout(regression_dropout),

            # Layer 2: d_fuse*3 → d_fuse*2
            nn.Linear(d_fuse * 3, d_fuse * 2, bias=False),
            nn.LayerNorm(d_fuse * 2),
            nn.GELU(),
            nn.Dropout(regression_dropout),

            # Layer 3: d_fuse*2 → d_fuse
            nn.Linear(d_fuse * 2, d_fuse, bias=False),
            nn.LayerNorm(d_fuse),
            nn.GELU(),
            nn.Dropout(regression_dropout * 0.7),  # Reduce dropout near output

            # Layer 4: d_fuse → d_fuse//2
            nn.Linear(d_fuse, d_fuse // 2, bias=False),
            nn.LayerNorm(d_fuse // 2),
            nn.GELU(),
            nn.Dropout(regression_dropout * 0.5),

            # Layer 5: Output
            nn.Linear(d_fuse // 2, 21, bias=False)  # 21 bins for scores 0-10 (step 0.5)
        )

        # PHASE 2: Fallback regression head for concatenation mode (backward compatibility)
        self.reg_head_concat = nn.Sequential(
            nn.Linear(5 * d_fuse, 2 * d_fuse, bias=False),
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
        """
        Encode text without pooling

        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]

        Returns:
            hidden_states: [B, seq_len, text_hidden_dim] or None if input_ids is None
        """
        if input_ids is None:
            return None
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden_states = outputs.hidden_states

        # Average last k layers
        k = min(self.avg_last_k, len(all_hidden_states))
        if k == 1:
            hidden_states = all_hidden_states[-1]
        else:
            hidden_states = torch.stack(all_hidden_states[-k:], dim=0).mean(dim=0)

        hidden_states = hidden_states.float()
        return hidden_states  # [B, seq_len, text_hidden_dim]

    def encode_audio(self, audio):
        """
        Encode audio chunks using audio encoder (Wav2Vec2/Whisper) + trainable adapter

        Supports two pooling modes:
        1. Hierarchical (hierarchical_audio_pooling=True):
           - Splits chunks into num_parts groups
           - Pools each group separately (mean or attention-based)
           - Returns [B, num_parts, d_fuse] (e.g., [B, 3, d_fuse])
           - Preserves part-level structure for cross-attention
           - Enhanced mode (use_enhanced_hierarchical=True): Uses PartAttentionPooling

        2. Flat (hierarchical_audio_pooling=False):
           - Returns all chunks unpooled
           - Returns [B, num_chunks, d_fuse]

        Args:
            audio: [B, num_chunks, waveform_len] for Wav2Vec2
                   [B, num_chunks, 80, 3000] for Whisper (log-mel)

        Returns:
            audio_features: [B, num_parts, d_fuse] if hierarchical, else [B, num_chunks, d_fuse]
        """
        if audio is None:
            print("⚠️ Warning: Audio input is None, skipping audio encoding.")
            return None
        # else:
        #     print("✓ Audio input received for encoding.")

        device = next(self.parameters()).device

        # Reshape based on audio dimensionality (auto-detect format)
        if audio.dim() == 3:
            # Wav2Vec2: [B, num_chunks, waveform_len] → [B*num_chunks, waveform_len]
            batch_size, num_chunks, waveform_len = audio.shape
            audio_flat = audio.view(batch_size * num_chunks, waveform_len).to(device)
        elif audio.dim() == 4:
            # Whisper: [B, num_chunks, mel_bins, time_steps] → [B*num_chunks, mel_bins, time_steps]
            batch_size, num_chunks, mel_bins, time_steps = audio.shape
            audio_flat = audio.view(batch_size * num_chunks, mel_bins, time_steps).to(device)
        else:
            raise ValueError(f"Unexpected audio dimensionality: {audio.dim()}. Expected 3D (Wav2Vec2) or 4D (Whisper)")

        # Encoder forward (trainable when not frozen)
        # train_audio_encoder = (not self.audio_encoder_frozen) or self.audio_encoder_use_lora
        # with torch.set_grad_enabled(train_audio_encoder):
            # NEW: Use abstracted encoder (supports Wav2Vec2 or Whisper)
        audio_features_flat = self.audio_encoder.forward(audio_flat)  # [B*chunks, seq_len, hidden_dim]
        audio_features_flat = audio_features_flat.mean(dim=1)  # [B*chunks, hidden_dim] (768 or 1280)

        # Apply trainable adapter
        audio_features_flat = self.audio_adapter(audio_features_flat)

        # Reshape back to [batch, chunks, hidden]
        audio_features = audio_features_flat.view(batch_size, num_chunks, -1)

        # Project to fusion space
        audio_features = self.audio_proj(audio_features)
        audio_features = self.audio_norm(audio_features)  # [B, num_chunks, d_fuse]

        if self.hierarchical_audio_pooling:
            # Hierarchical pooling: split into parts and pool each separately
            chunks_per_part = num_chunks // self.num_parts
            part_features = []

            for i in range(self.num_parts):
                start_idx = i * chunks_per_part
                end_idx = start_idx + chunks_per_part
                part_chunks = audio_features[:, start_idx:end_idx, :]  # [B, chunks_per_part, d_fuse]

                # NEW: Add chunk position embeddings (if enhanced hierarchical enabled)
                if self.use_enhanced_hierarchical and hasattr(self, 'chunk_position_embeds'):
                    part_chunks = part_chunks + self.chunk_position_embeds.unsqueeze(0)  # [B, chunks_per_part, d_fuse]

                # NEW: Use attention pooling or mean pooling
                if self.use_enhanced_hierarchical and hasattr(self, 'part_attn_pooling'):
                    part_pooled = self.part_attn_pooling(part_chunks)  # Learned attention weights
                else:
                    part_pooled = part_chunks.mean(dim=1)  # Legacy mean pooling

                part_features.append(part_pooled)

            # Stack parts as separate tokens: [B, num_parts, d_fuse]
            audio_features = torch.stack(part_features, dim=1)  # [B, 3, d_fuse]

            # Add part position embeddings (distinguish Part 1, 2, 3)
            part_embeds = self.part_embed_proj(self.part_position_embeds)  # [num_parts, d_fuse]
            audio_features = audio_features + part_embeds.unsqueeze(0)  # Broadcast: [B, num_parts, d_fuse]

        return audio_features  # [B, num_parts, d_fuse] or [B, num_chunks, d_fuse]

    def forward(self, input_ids=None, attention_mask=None, audio=None,
                question_input_ids=None, question_attention_mask=None,
                response_input_ids=None, response_attention_mask=None):
        """
        Forward pass with optional question-aware encoding (STEP 3)

        When use_question_encoder=False (standard mode):
            - Uses input_ids and attention_mask (Q+R concatenated)
            - Standard flow unchanged

        When use_question_encoder=True (question-aware mode):
            - Uses question_input_ids, question_attention_mask (questions only)
            - Uses response_input_ids, response_attention_mask (responses only)
            - Encodes separately, then applies cross-attention

        Args:
            input_ids: [B, seq_len] - Full text (Q+R concat), used when use_question_encoder=False
            attention_mask: [B, seq_len] - Mask for full text
            audio: [B, num_chunks, waveform_len]
            question_input_ids: [B, Q_len] - Questions only, used when use_question_encoder=True
            question_attention_mask: [B, Q_len] - Mask for questions
            response_input_ids: [B, R_len] - Responses only, used when use_question_encoder=True
            response_attention_mask: [B, R_len] - Mask for responses

        Returns:
            expected_score: [B] - Predicted scores
        """
        # Encode audio first (needed for both modes)
        audio_features = self.encode_audio(audio)  # [B, num_chunks, d_fuse] or None

        # STEP 3: Question-aware or standard text encoding
        if self.use_question_encoder and question_input_ids is not None:
            # NEW: Question-aware encoding with cross-attention

            # 1. Encode question
            question_hidden = self.encode_text(question_input_ids, question_attention_mask)
            question_features = self.text_proj(question_hidden)      # [B, Q_len, d_fuse]
            question_features = self.text_norm(question_features)

            # 2. Encode response
            response_hidden = self.encode_text(response_input_ids, response_attention_mask)
            response_features = self.text_proj(response_hidden)      # [B, R_len, d_fuse]
            response_features = self.text_norm(response_features)

            # 3. Question-aware encoding with cross-attention
            text_self_pooled = self.question_aware_encoder(
                question_features,
                response_features,
                question_attention_mask,
                response_attention_mask
            )  # [B, d_fuse]

            # Use response_features for cross-attention with audio
            text_features = response_features
            attention_mask = response_attention_mask

        else:
            # ORIGINAL: Standard text encoding (backward compatible)
            text_hidden = self.encode_text(input_ids, attention_mask)  # [B, seq_len, text_hidden] or None

            if text_hidden is not None:
                print("✓ Text input received for encoding.")
                # Project text to fusion space
                text_features = self.text_proj(text_hidden)  # [B, seq_len, d_fuse]
                text_features = self.text_norm(text_features)

                # 1. Text Self-Attention
                if self.num_self_attn_layers > 1:
                    # PHASE 2: Multi-layer self-attention (includes residual + norm internally)
                    text_self_out = self.text_self_attention(
                        text_features,
                        mask=(attention_mask == 0)
                    )
                else:
                    # Original: Single-layer attention with external residual + norm
                    text_self_out, _ = self.text_self_attention(
                        text_features, text_features, text_features,
                        key_padding_mask=(attention_mask == 0)
                    )
                    text_self_out = self.text_self_norm(text_features + text_self_out)

                # Pool text self-attention output
                text_self_pooled = self.text_self_pool(text_self_out, attention_mask)  # [B, d_fuse]
            else:
                # Text is None - will skip text-related processing
                text_features = None
                text_self_pooled = None

        # ========== ATTENTION FUSION ==========

        if audio_features is not None:
            # 2. Audio Self-Attention
            if self.num_self_attn_layers > 1:
                # PHASE 2: Multi-layer self-attention (includes residual + norm internally)
                audio_self_out = self.audio_self_attention(
                    audio_features,
                    mask=None  # No mask needed - audio has no padding
                )
            else:
                # Original: Single-layer attention with external residual + norm
                audio_self_out, _ = self.audio_self_attention(
                    audio_features, audio_features, audio_features
                    # No mask needed - audio has no padding
                )
                audio_self_out = self.audio_self_norm(audio_features + audio_self_out)  # [B, num_parts, d_fuse]

            audio_self_pooled = self.audio_self_pool(audio_self_out)  # [B, d_fuse]

            # 3. Text-to-Audio Cross-Attention (q=text, kv=audio) - skip if text is None
            if text_features is not None:
                t2a_out, _ = self.text_to_audio_attention(
                    text_features, audio_features, audio_features
                )
                t2a_out = self.t2a_norm(text_features + t2a_out)
                t2a_pooled = self.t2a_pool(t2a_out, attention_mask)  # [B, d_fuse]

                # 4. Audio-to-Text Cross-Attention (q=audio, kv=text)
                a2t_out, _ = self.audio_to_text_attention(
                    audio_features, text_features, text_features,
                    key_padding_mask=(attention_mask == 0)
                )
                a2t_out = self.a2t_norm(audio_features + a2t_out)  # [B, num_parts/num_chunks, d_fuse]
            else:
                # Text is None - skip cross-attention
                t2a_pooled = None
                a2t_out = audio_features  # Use original audio features

            # Apply part importance weights (only for hierarchical mode)
            if self.hierarchical_audio_pooling:
                # Normalize importance weights to sum to 1
                importance = F.softmax(self.part_importance, dim=0)  # [num_parts]
                # Weight each part: [B, num_parts, d_fuse] * [1, num_parts, 1]
                a2t_out = a2t_out * importance.view(1, -1, 1)

            a2t_pooled = self.a2t_pool(a2t_out)  # [B, d_fuse]

            # NEW: Part-level mean pooling (global audio representation)
            audio_mean_pooled = audio_features.mean(dim=1)  # [B, d_fuse]

            # ========== MULTIMODAL FUSION ==========
            if text_self_pooled is None:
                # Audio only (no text) - use zero-padded text features
                if self.use_gated_fusion:
                    fused = self.gated_fusion(
                        torch.zeros_like(audio_self_pooled),   # [B, d_fuse] - text self-attention (missing)
                        audio_self_pooled,     # [B, d_fuse] - audio self-attention
                        torch.zeros_like(audio_self_pooled),   # [B, d_fuse] - text→audio (missing)
                        a2t_pooled,            # [B, d_fuse] - audio→text (no text, just audio)
                        audio_mean_pooled      # [B, d_fuse] - global audio
                    )  # [B, d_fuse]
                else:
                    fused = torch.cat([
                        torch.zeros_like(audio_self_pooled),   # [B, d_fuse] - text self-attention (missing)
                        audio_self_pooled,     # [B, d_fuse] - audio self-attention
                        torch.zeros_like(audio_self_pooled),   # [B, d_fuse] - text→audio (missing)
                        a2t_pooled,            # [B, d_fuse] - audio→text (no text, just audio)
                        audio_mean_pooled      # [B, d_fuse] - global audio
                    ], dim=-1)  # [B, 5*d_fuse]
            elif self.use_gated_fusion:
                # PHASE 2: Gated fusion with learned modality interactions
                fused = self.gated_fusion(
                    text_self_pooled,      # [B, d_fuse] - text self-attention
                    audio_self_pooled,     # [B, d_fuse] - audio self-attention
                    t2a_pooled,            # [B, d_fuse] - text→audio cross-attention
                    a2t_pooled,            # [B, d_fuse] - audio→text cross-attention
                    audio_mean_pooled      # [B, d_fuse] - global audio
                )  # [B, d_fuse]
            else:
                # Original: Simple concatenation
                fused = torch.cat([
                    text_self_pooled,      # [B, d_fuse] - text self-attention
                    audio_self_pooled,     # [B, d_fuse] - audio self-attention
                    t2a_pooled,            # [B, d_fuse] - text→audio cross-attention
                    a2t_pooled,            # [B, d_fuse] - audio→text cross-attention
                    audio_mean_pooled      # [B, d_fuse] - global audio
                ], dim=-1)  # [B, 5*d_fuse]
        else:
            # Text only (fallback) - pad with zeros for missing audio features
            if self.use_gated_fusion:
                # PHASE 2: Use gated fusion with zero-padded audio features
                fused = self.gated_fusion(
                    text_self_pooled,                     # [B, d_fuse] - text self-attention
                    torch.zeros_like(text_self_pooled),   # [B, d_fuse] - audio self-attention (missing)
                    torch.zeros_like(text_self_pooled),   # [B, d_fuse] - text→audio (missing)
                    torch.zeros_like(text_self_pooled),   # [B, d_fuse] - audio→text (missing)
                    torch.zeros_like(text_self_pooled)    # [B, d_fuse] - audio mean (missing)
                )  # [B, d_fuse]
            else:
                # Original: Simple concatenation with zero-padded audio
                fused = torch.cat([
                    text_self_pooled,                     # [B, d_fuse] - text self-attention
                    torch.zeros_like(text_self_pooled),   # [B, d_fuse] - audio self-attention (missing)
                    torch.zeros_like(text_self_pooled),   # [B, d_fuse] - text→audio (missing)
                    torch.zeros_like(text_self_pooled),   # [B, d_fuse] - audio→text (missing)
                    torch.zeros_like(text_self_pooled)    # [B, d_fuse] - audio mean (missing)
                ], dim=-1)  # [B, 5*d_fuse]

        # ========== REGRESSION HEAD ==========
        if self.use_gated_fusion:
            # PHASE 2: Deeper 5-layer regression head (input: [B, d_fuse])
            logits = self.reg_head(fused)  # [B, 21]
        else:
            # Original: 3-layer regression head (input: [B, 5*d_fuse])
            logits = self.reg_head_concat(fused)  # [B, 21]
        probs = F.softmax(logits, dim=-1)

        # Expected score (0-10)
        score_bins = torch.linspace(0, 10, steps=21, device=logits.device)
        expected_score = (probs * score_bins).sum(dim=-1)

        return {
            'logits': logits,           # [B, 21]
            'probs': probs,             # [B, 21]
            'expected_score': expected_score  # [B]
        }

    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'd_fuse': self.d_fuse,
            'num_parts': self.num_parts
        }, path)

    @classmethod
    def load(cls, path, **kwargs):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(d_fuse=checkpoint.get('d_fuse', 1024), **kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


if __name__ == "__main__":
    """
    Test script for ESL Grading Model
    Usage: python model.py
    """
    import yaml
    from pathlib import Path

    print("=" * 80)
    print("Testing ESL Grading Model with Audio")
    print("=" * 80)

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"\n✓ Config loaded from {config_path}")
    print(f"  Audio encoder: {cfg['model']['audio_encoder_type']}")
    print(f"  d_fuse: {cfg['model']['d_fuse']}")
    print(f"  Hierarchical pooling: {cfg['model']['hierarchical_audio_pooling']}")
    print(f"  Enhanced hierarchical: {cfg['model']['use_enhanced_hierarchical']}")

    # Create model with config
    model = ESLGradingModelByCandidatesWithAudio(
        model_name=cfg['model']['model_name'],
        audio_encoder_id=cfg['model']['audio_encoder_id'],
        audio_encoder_type=cfg['model']['audio_encoder_type'],
        audio_encoder_frozen=cfg['model']['audio_encoder_frozen'],
        d_fuse=cfg['model']['d_fuse'],
        pooling_dropout=cfg['model']['pooling_dropout'],
        regression_dropout=cfg['model']['regression_dropout'],
        avg_last_k=cfg['model']['avg_last_k'],
        adapter_bottleneck_dim=cfg['model']['adapter_bottleneck_dim'],
        adapter_num_heads=cfg['model']['adapter_num_heads'],
        adapter_dropout=cfg['model']['adapter_dropout'],
        num_parts=cfg['model']['num_parts'],
        hierarchical_audio_pooling=cfg['model']['hierarchical_audio_pooling'],
        use_enhanced_hierarchical=cfg['model']['use_enhanced_hierarchical'],
    )

    print(f"\n✓ Model created successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nParameter Summary:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen (audio encoder): {frozen_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Test forward pass
    print(f"\n{'='*80}")
    print("Testing Forward Pass")
    print("=" * 80)

    batch_size = 2
    seq_len = 100
    num_chunks = cfg['audio']['num_chunks'] * cfg['model']['num_parts']  # 6 chunks × 3 parts = 18

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Create dummy audio based on encoder type
    if cfg['model']['audio_encoder_type'].lower() == 'wav2vec2':
        # Wav2Vec2: raw waveform [B, num_chunks, waveform_len]
        waveform_len = cfg['audio']['max_waveform_len']  # 30s × 16kHz = 480000
        audio = torch.randn(batch_size, num_chunks, waveform_len)
        print(f"\nInput shapes:")
        print(f"  Text: [batch={batch_size}, seq_len={seq_len}]")
        print(f"  Audio (Wav2Vec2): [batch={batch_size}, chunks={num_chunks}, waveform={waveform_len}]")
    else:  # whisper
        # Whisper: log-mel spectrogram [B, num_chunks, num_mel_bins, time_steps]
        # Get mel bins from encoder (80 for base/small, 128 for large/turbo)
        mel_bins = model.audio_encoder.get_num_mel_bins()
        time_steps = 3000  # Fixed for Whisper (30s chunks)
        audio = torch.randn(batch_size, num_chunks, mel_bins, time_steps)
        print(f"\nInput shapes:")
        print(f"  Text: [batch={batch_size}, seq_len={seq_len}]")
        print(f"  Audio (Whisper): [batch={batch_size}, chunks={num_chunks}, mel={mel_bins}, time={time_steps}]")
        print(f"  Note: Whisper {cfg['model']['audio_encoder_id']} uses {mel_bins} mel bins")

    print(f"\nRunning forward pass...")
    outputs = model(input_ids, attention_mask, audio)

    print(f"\n✓ Forward pass successful!")
    print(f"\nOutput shapes:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Probs: {outputs['probs'].shape}")
    print(f"  Expected scores: {outputs['expected_score'].shape}")

    print(f"\nSample predictions:")
    for i in range(batch_size):
        score = outputs['expected_score'][i].item()
        print(f"  Sample {i+1}: {score:.2f}/10.0")

    print(f"\n{'='*80}")
    print("✅ All tests passed!")
    print("=" * 80)
