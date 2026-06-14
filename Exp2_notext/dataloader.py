"""
Data Loading for ESL Speaking Grading Model

Includes:
- ESLDatasetByCandidatesWithAudio: Dataset with audio and text
- InverseScoreSampler: Inverse-frequency sampling for imbalanced data
- StratifiedScoreSampler: Ensure batch contains both edge and middle scores
- Collate functions with fixed padding
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
import numpy as np
import ast
from collections import Counter
from utils import _process_audio_file, clean_dataframe_bycandidates


# ============================================================================
# Dataset Class
# ============================================================================

class ESLDatasetByCandidatesWithAudio(Dataset):
    """
    Enhanced ESL Dataset supporting both text and audio with question prompts

    Each candidate has 3 parts with questions and responses
    """

    def __init__(self, dataframe, criteria='final', audio_processor=None,
                 encoder_type='wav2vec2', remove_low_content=True, num_chunks=6, chunk_length_sec=30,
                 separate_question_response=False):  # STEP 3: NEW parameter
        """
        Args:
            dataframe: DataFrame with columns 'text', scoring columns, 'question_type', 'absolute_paths', 'Question'
            criteria: Scoring criterion to use ('final', 'grammar', 'vocabulary', etc.)
            audio_processor: Wav2Vec2Processor or WhisperProcessor instance
            encoder_type: "wav2vec2" or "whisper" (NEW)
            remove_low_content: Whether to remove low content samples
            num_chunks: Number of audio chunks to extract (default 6)
            chunk_length_sec: Length of each audio chunk in seconds (default 30s)
            separate_question_response: Whether to return separate questions and responses (STEP 3)
        """
        self.criteria = criteria
        self.audio_processor = audio_processor
        self.encoder_type = encoder_type  # NEW
        self.num_chunks = num_chunks
        self.chunk_length_sec = chunk_length_sec
        self.separate_question_response = separate_question_response  # STEP 3

        # Candidate IDs
        self.candidate_ids = dataframe['Candidate_ID'].tolist()

        # Prompts
        self.text_prefix = f"The following is a spoken English response by a non-native speaker. Grade the {criteria} score based on the transcript below:"
        self.question_prefix = "Question: "

        # Question type descriptions
        self.question_type_map = {
            1: "Social Interaction: Answer several questions about familiar topics",
            2: "Solution Discussion: Choose one option from a situation and justify your choice",
            3: "Topic Development: Present a given topic with supporting ideas and answer follow-up questions"
        }

        # Question types and scores
        self.question_types = dataframe['question_type'].apply(ast.literal_eval).tolist()
        self.scores = dataframe[criteria].astype(float).tolist()

        # Parse questions if available
        if 'Question' in dataframe.columns:
            print("✓ Questions column found, parsing questions.")
            self.questions = dataframe['Question'].apply(ast.literal_eval).tolist()
        else:
            self.questions = None

        # Process texts with questions
        raw_texts = dataframe['text'].apply(ast.literal_eval).tolist()
        self.raw_responses = raw_texts  # STEP 3: Store raw responses
        self.texts = []

        # NOTEXT EXPERIMENT: Use fixed placeholder instead of actual transcript
        placeholder_text = "This is work of candidate: "

        for idx, (raw_text, question_types) in enumerate(zip(raw_texts, self.question_types)):
            candidate_texts = []

            for i, (t, qtype) in enumerate(zip(raw_text, question_types)):
                # Use placeholder text instead of actual transcript
                formatted_text = placeholder_text
                candidate_texts.append(formatted_text)

            self.texts.append(candidate_texts)

        # Audio paths
        self.absolute_paths = (
            dataframe['absolute_paths'].apply(ast.literal_eval).tolist()
            if 'absolute_paths' in dataframe.columns
            else [None] * len(self.texts)
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            'Candidate_ID': self.candidate_ids[idx],
            'text': self.texts[idx],  # List of 3 formatted texts with questions
            'score': torch.tensor(self.scores[idx], dtype=torch.float32),
            'question_type': self.question_types[idx],
            'absolute_path': self.absolute_paths[idx]
        }

        # STEP 3: Add separate questions and responses if requested
        if self.separate_question_response:
            formatted_questions = []
            formatted_responses = []

            for i in range(len(self.raw_responses[idx])):
                # Format question
                if self.questions is not None and idx < len(self.questions):
                    question_text = self.questions[idx][i]
                else:
                    question_text = self.question_type_map.get(self.question_types[idx][i], '')

                formatted_questions.append(f"{self.question_prefix}{question_text}")

                # Format response
                response_text = self.raw_responses[idx][i]
                formatted_responses.append(f"{self.text_prefix}\n{response_text}")

            item['questions'] = formatted_questions  # List of 3 formatted questions
            item['responses'] = formatted_responses  # List of 3 formatted responses

        # Process audio if available
        if self.absolute_paths[idx] is not None and self.audio_processor is not None:
            item['audio'] = []
            item['has_audio'] = []

            for absolute_path in self.absolute_paths[idx]:
                try:
                    audio_tensor = _process_audio_file(
                        absolute_path,
                        self.audio_processor,
                        encoder_type=self.encoder_type,  # NEW
                        num_chunks=self.num_chunks,
                        chunk_length_sec=self.chunk_length_sec
                    )
                    item['audio'].append(audio_tensor)
                    item['has_audio'].append(True)
                except Exception as e:
                    print(f"Error processing audio {absolute_path}: {e}")
                    # Create dummy audio tensor if processing fails
                    chunk_samples = int(self.chunk_length_sec * 16000)
                    item['audio'].append(torch.zeros(self.num_chunks, chunk_samples))
                    item['has_audio'].append(False)
        else:
            # Create dummy audio tensor
            chunk_samples = int(self.chunk_length_sec * 16000)
            item['audio'] = [torch.zeros(self.num_chunks, chunk_samples)]
            item['has_audio'] = [False]

        return item


# ============================================================================
# Samplers
# ============================================================================

class InverseScoreSampler(Sampler):
    """
    Inverse-frequency sampling to balance imbalanced score distribution

    Over-samples minority scores, under-samples majority scores
    """

    def __init__(self, dataset, alpha=0.2, replacement=True):
        """
        Args:
            dataset: ESL Dataset
            alpha: Weighting power (0=random, 1=pure inverse-frequency)
            replacement: Sample with replacement
        """
        self.dataset = dataset
        self.replacement = replacement
        self.alpha = alpha

        # Round scores to nearest 0.5 for binning
        binned_scores = [round(float(s) * 2) / 2 for s in dataset.scores]
        counter = Counter(binned_scores)

        # Compute inverse frequency weights
        freqs = np.array([counter[round(float(s) * 2) / 2] for s in dataset.scores], dtype=np.float32)
        self.weights = (1.0 / freqs) ** alpha
        self.weights /= self.weights.sum()  # Normalize to sum to 1

    def __iter__(self):
        n = len(self.dataset)
        indices = np.random.choice(
            np.arange(n), size=n, replace=self.replacement, p=self.weights
        )
        return iter(indices.tolist())

    def __len__(self):
        return len(self.dataset)


class StratifiedScoreSampler(Sampler):
    """
    Ensure each batch has representation from both edge and middle scores

    Edge scores: <= edge_threshold or >= (10 - edge_threshold)
    Middle scores: between edge thresholds
    """

    def __init__(self, dataset, edge_threshold=3.5, edge_ratio=0.3):
        """
        Args:
            dataset: ESL Dataset
            edge_threshold: Threshold for edge scores
            edge_ratio: Ratio of edge samples in each epoch
        """
        self.dataset = dataset
        self.edge_threshold = edge_threshold
        self.edge_ratio = edge_ratio

        # Split indices into edge and middle
        scores = np.array([float(s) for s in dataset.scores])
        self.edge_indices = np.where(
            (scores <= edge_threshold) | (scores >= (10 - edge_threshold))
        )[0].tolist()
        self.middle_indices = np.where(
            (scores > edge_threshold) & (scores < (10 - edge_threshold))
        )[0].tolist()

        print(f"Stratified Sampler: {len(self.edge_indices)} edge, {len(self.middle_indices)} middle samples")

    def __iter__(self):
        # Handle edge cases where one group is empty
        if len(self.edge_indices) == 0 and len(self.middle_indices) == 0:
            raise ValueError("Dataset has no valid samples!")

        if len(self.edge_indices) == 0:
            # No edge samples - sample all from middle
            print("⚠️  StratifiedSampler: No edge samples, using all middle samples")
            indices = np.random.choice(self.middle_indices, size=len(self.dataset), replace=True)
        elif len(self.middle_indices) == 0:
            # No middle samples - sample all from edge (e.g., edge-only datasets)
            print("⚠️  StratifiedSampler: No middle samples, using all edge samples")
            indices = np.random.choice(self.edge_indices, size=len(self.dataset), replace=True)
        else:
            # Normal case: both groups have samples
            n_edge = int(len(self.dataset) * self.edge_ratio)
            n_middle = len(self.dataset) - n_edge
            edge_samples = np.random.choice(self.edge_indices, size=n_edge, replace=True)
            middle_samples = np.random.choice(self.middle_indices, size=n_middle, replace=True)
            indices = np.concatenate([edge_samples, middle_samples])

        np.random.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return len(self.dataset)


# ============================================================================
# Collate Functions
# ============================================================================

def get_collate_fn_bycandidates_with_audio(tokenizer, max_length=8192,
                                            max_audio_chunks=30,
                                            max_waveform_len=288000,
                                            separate_tokenize=False):  # STEP 3: NEW parameter
    """
    Collate function with fixed padding for consistent tensor shapes (STEP 3: supports separate tokenization)

    Args:
        tokenizer: Hugging Face tokenizer
        max_length: Max text length
        max_audio_chunks: Max chunks per candidate (30 = 3 parts × 10 chunks)
        max_waveform_len: Fixed waveform length (288000 = 18s × 16kHz)
        separate_tokenize: Whether to tokenize questions and responses separately (STEP 3)

    Returns:
        Collate function
    """
    def collate_fn(batch):
        cand_texts = []
        cand_audios = []
        cand_IDs = []
        scores = []
        all_question_types = []

        # STEP 3: Collect questions and responses separately if needed
        cand_questions = [] if separate_tokenize else None
        cand_responses = [] if separate_tokenize else None

        for item in batch:
            # ----------- Text part -------------
            if separate_tokenize:
                # STEP 3: Separate tokenization mode
                # Join 3 parts with [SEP]
                cand_questions.append(" [SEP] ".join(item['questions']))
                cand_responses.append(" [SEP] ".join(item['responses']))
            else:
                # Original: Concatenated Q+R
                # Format: Q1 + R1 [SEP] Q2 + R2 [SEP] Q3 + R3
                cand_texts.append(" [SEP] ".join(item['text']))

            all_question_types.extend(item['question_type'])

            # ----------- Audio part -------------
            chunks = [a if torch.is_tensor(a) else torch.tensor(a) for a in item['audio']]
            cand_audio = torch.cat(chunks, dim=0)  # [num_chunks_total, waveform_length]
            cand_audios.append(cand_audio)

            # ----------- Label -----------
            scores.append(item['score'])

            # ----------- Candidate ID------------
            cand_IDs.append(item['Candidate_ID'])

        # STEP 3: Tokenize text (separate or concatenated)
        if separate_tokenize:
            # Tokenize questions (shorter)
            question_encoded = tokenizer(
                cand_questions,
                padding='max_length',
                truncation=True,
                max_length=512,  # Questions are shorter
                return_tensors='pt'
            )

            # Tokenize responses (longer)
            response_encoded = tokenizer(
                cand_responses,
                padding='max_length',
                truncation=True,
                max_length=max_length - 512,  # Remaining for responses
                return_tensors='pt'
            )
        else:
            # Original: Concatenated tokenization
            encoded = tokenizer(
                cand_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

        # Fixed-size audio padding (handle both Wav2Vec2 and Whisper)
        padded_audios = []

        for a in cand_audios:
            if a is None or a.numel() == 0:
                # No audio: fill with zeros
                # Default to Wav2Vec2 format [chunks, waveform_len]
                padded = torch.zeros((max_audio_chunks, max_waveform_len), dtype=torch.float)
            else:
                # Detect audio format by number of dimensions
                if a.dim() == 2:
                    # Wav2Vec2: [chunks, waveform_len]
                    C, L = a.shape
                    pad_C = max(0, max_audio_chunks - C)
                    pad_L = max(0, max_waveform_len - L)
                    padded = F.pad(a, (0, pad_L, 0, pad_C), value=0.0)
                    padded = padded[:max_audio_chunks, :max_waveform_len]

                elif a.dim() == 3:
                    # Whisper: [chunks, mel_bins, time_steps]
                    C = a.shape[0]
                    pad_C = max(0, max_audio_chunks - C)
                    # No padding for mel_bins (fixed by processor)
                    # No padding for time_steps (fixed at 3000)
                    padded = F.pad(a, (0, 0, 0, 0, 0, pad_C), value=0.0)
                    padded = padded[:max_audio_chunks, :, :]
                else:
                    raise ValueError(f"Unexpected audio shape: {a.shape}")

            padded_audios.append(padded)

        audio_tensor = torch.stack(padded_audios, dim=0)  # [B, max_chunks, ...] (2D or 3D)
        score_tensor = torch.stack(scores) if isinstance(scores[0], torch.Tensor) else torch.tensor(scores, dtype=torch.float)

        # STEP 3: Build return dict with conditional fields
        result = {
            # Common fields
            'question_type': torch.tensor(all_question_types, dtype=torch.long),
            'audio': audio_tensor,                             # [B, max_chunks, max_waveform]
            'score': score_tensor,                             # [B]
            'candidate_id': cand_IDs,                          # [B]
            "absolute_path": [item["absolute_path"] for item in batch]
        }

        if separate_tokenize:
            # STEP 3: Separate tokenization
            result.update({
                'question_input_ids': question_encoded['input_ids'],              # [B, 512]
                'question_attention_mask': question_encoded['attention_mask'],    # [B, 512]
                'response_input_ids': response_encoded['input_ids'],              # [B, 7680]
                'response_attention_mask': response_encoded['attention_mask'],    # [B, 7680]
            })
        else:
            # Original: Concatenated tokenization
            result.update({
                'input_ids': encoded['input_ids'],                 # [B, T_text]
                'attention_mask': encoded['attention_mask'],       # [B, T_text]
            })

        return result

    return collate_fn

def get_collate_fn_bycandidates_without_text(tokenizer, max_length=8192,
                                            max_audio_chunks=30,
                                            max_waveform_len=288000):  # STEP 3: NEW parameter
    """
    Collate function using only audio, no text (STEP 3: supports separate tokenization)

    Args:
        tokenizer: Hugging Face tokenizer
        max_length: Max text length
        max_audio_chunks: Max chunks per candidate (30 = 3 parts × 10 chunks)
        max_waveform_len: Fixed waveform length (288000 = 18s × 16kHz)
        separate_tokenize: Whether to tokenize questions and responses separately (STEP 3)

    Returns:
        Collate function
    """
    def collate_fn(batch):
        cand_texts = []
        cand_audios = []
        cand_IDs = []
        scores = []
        all_question_types = []

        for item in batch:
            # # ----------- Text part -------------
            # if separate_tokenize:
            #     # STEP 3: Separate tokenization mode
            #     # Join 3 parts with [SEP]
            #     cand_questions.append(" [SEP] ".join(item['questions']))
            #     cand_responses.append(" [SEP] ".join(item['responses']))
            # else:
            #     # Original: Concatenated Q+R
            #     # Format: Q1 + R1 [SEP] Q2 + R2 [SEP] Q3 + R3
            #     cand_texts.append(" [SEP] ".join(item['text']))

            cand_texts.append("This is work of candidate: ")  # Placeholder empty text

            all_question_types.extend(item['question_type'])

            # ----------- Audio part -------------
            chunks = [a if torch.is_tensor(a) else torch.tensor(a) for a in item['audio']]
            cand_audio = torch.cat(chunks, dim=0)  # [num_chunks_total, waveform_length]
            cand_audios.append(cand_audio)

            # ----------- Label -----------
            scores.append(item['score'])

            # ----------- Candidate ID------------
            cand_IDs.append(item['Candidate_ID'])
        # Original: Concatenated tokenization
        encoded = tokenizer(
            cand_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Fixed-size audio padding (handle both Wav2Vec2 and Whisper)
        padded_audios = []

        for a in cand_audios:
            if a is None or a.numel() == 0:
                # No audio: fill with zeros
                # Default to Wav2Vec2 format [chunks, waveform_len]
                padded = torch.zeros((max_audio_chunks, max_waveform_len), dtype=torch.float)
            else:
                # Detect audio format by number of dimensions
                if a.dim() == 2:
                    # Wav2Vec2: [chunks, waveform_len]
                    C, L = a.shape
                    pad_C = max(0, max_audio_chunks - C)
                    pad_L = max(0, max_waveform_len - L)
                    padded = F.pad(a, (0, pad_L, 0, pad_C), value=0.0)
                    padded = padded[:max_audio_chunks, :max_waveform_len]

                elif a.dim() == 3:
                    # Whisper: [chunks, mel_bins, time_steps]
                    C = a.shape[0]
                    pad_C = max(0, max_audio_chunks - C)
                    # No padding for mel_bins (fixed by processor)
                    # No padding for time_steps (fixed at 3000)
                    padded = F.pad(a, (0, 0, 0, 0, 0, pad_C), value=0.0)
                    padded = padded[:max_audio_chunks, :, :]
                else:
                    raise ValueError(f"Unexpected audio shape: {a.shape}")

            padded_audios.append(padded)

        audio_tensor = torch.stack(padded_audios, dim=0)  # [B, max_chunks, ...] (2D or 3D)
        score_tensor = torch.stack(scores) if isinstance(scores[0], torch.Tensor) else torch.tensor(scores, dtype=torch.float)

        # STEP 3: Build return dict with conditional fields
        result = {
            # Common fields
            'question_type': torch.tensor(all_question_types, dtype=torch.long),
            'audio': audio_tensor,                             # [B, max_chunks, max_waveform]
            'score': score_tensor,                             # [B]
            'candidate_id': cand_IDs,                          # [B]
            "absolute_path": [item["absolute_path"] for item in batch]
        }

        # Original: Concatenated tokenization
        result.update({
            'input_ids': encoded['input_ids'],                 # [B, T_text]
            'attention_mask': encoded['attention_mask'],       # [B, T_text]
        })

        return result

    return collate_fn


if __name__ == "__main__":
    """
    Test dataloader and collate_fn with both Wav2Vec2 and Whisper audio formats
    Usage: python dataloader.py
    """
    import torch
    from transformers import AutoTokenizer

    print("=" * 80)
    print("Testing Dataloader & Collate Function")
    print("=" * 80)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)

    # Test collate function
    print("\n1. Testing collate_fn with Wav2Vec2 audio (2D):")
    collate = get_collate_fn_bycandidates_without_text(tokenizer, max_length=512, max_audio_chunks=18, max_waveform_len=480000)

    # Create dummy batch with Wav2Vec2 audio [chunks, waveform_len]
    batch_wav2vec2 = [
        {
            'text': ["", "", ""],  # Dummy text
            'audio': [
                torch.randn(6, 480000),  # Part 1: 6 chunks
                torch.randn(6, 480000),  # Part 2: 6 chunks
                torch.randn(6, 480000),  # Part 3: 6 chunks
            ],
            'score': 7.5,
            'question_type': [1, 2, 3],
            'Candidate_ID': 'TEST001',
            'absolute_path': ['path1.wav', 'path2.wav', 'path3.wav']
        },
        {
            'text': ["", "", ""],  # Dummy text
            'audio': [
                torch.randn(6, 480000),
                torch.randn(6, 480000),
                torch.randn(6, 480000),
            ],
            'score': 6.0,
            'question_type': [1, 2, 3],
            'Candidate_ID': 'TEST002',
            'absolute_path': ['path4.wav', 'path5.wav', 'path6.wav']
        }
    ]

    result = collate(batch_wav2vec2)
    print(f"   ✓ Input IDs shape: {result['input_ids'].shape}")
    print(f"   ✓ Audio shape: {result['audio'].shape}")
    print(f"   ✓ Expected: [batch=2, chunks=18, waveform=480000]")
    assert result['audio'].dim() == 3, "Wav2Vec2 audio should be 3D"

    # Test collate function with Whisper audio
    print("\n2. Testing collate_fn with Whisper audio (3D):")

    # Create dummy batch with Whisper audio [chunks, mel_bins, time_steps]
    batch_whisper = [
        {
            'text': ["", "", ""],  # Dummy text
            'audio': [
                torch.randn(6, 128, 3000),  # Part 1: 6 chunks, 128 mel bins, 3000 time steps
                torch.randn(6, 128, 3000),  # Part 2
                torch.randn(6, 128, 3000),  # Part 3
            ],
            'score': 8.0,
            'question_type': [1, 2, 3],
            'Candidate_ID': 'TEST003',
            'absolute_path': ['path7.wav', 'path8.wav', 'path9.wav']
        },
        {
            'text': ["", "", ""],  # Dummy text
            'audio': [
                torch.randn(6, 128, 3000),
                torch.randn(6, 128, 3000),
                torch.randn(6, 128, 3000),
            ],
            'score': 7.0,
            'question_type': [1, 2, 3],
            'Candidate_ID': 'TEST004',
            'absolute_path': ['path10.wav', 'path11.wav', 'path12.wav']
        }
    ]

    result = collate(batch_whisper)
    print(f"   ✓ Input IDs shape: {result['input_ids'].shape}")
    print(f"   ✓ Audio shape: {result['audio'].shape}")
    print(f"   ✓ Expected: [batch=2, chunks=18, mel=128, time=3000]")
    assert result['audio'].dim() == 4, "Whisper audio should be 4D"

    print("\n" + "=" * 80)
    print("✅ All collate_fn tests passed!")
    print("   - Wav2Vec2 format: [B, chunks, waveform_len]")
    print("   - Whisper format: [B, chunks, mel_bins, time_steps]")
    print("=" * 80)

