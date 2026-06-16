import re
from collections import Counter
from typing import List, Set

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from nltk.corpus import stopwords

    NLTK_STOPWORDS: Set[str] = set(stopwords.words("english"))
except LookupError:
    NLTK_STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "i", "you", "we", "they",
    }

EXTRA_FILLERS = {
    "um", "uh", "hmm", "mmm", "ah", "oh",
    "...", "--", ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]",
    "{", "}", "\"", "'", "/", "\\",
}

ALL_STOPWORDS = NLTK_STOPWORDS.union(EXTRA_FILLERS)


def tokenize(text: str) -> List[str]:
    """Extract lower-cased alphabetic tokens and bracketed ASR annotations."""
    toks = re.findall(r"\[[^\]]+\]|[A-Za-z]+", str(text))
    return [tok.lower() for tok in toks]


def count_content_words(text: str, stopwords_: Set[str] = ALL_STOPWORDS) -> int:
    return sum(1 for tok in tokenize(text) if tok not in stopwords_)


def is_low_content(transcript: str, threshold: int = 5) -> bool:
    return count_content_words(transcript) < threshold


def replace_repeats(text: str, k: int = 3, tag: str = "") -> str:
    """Collapse contiguous token sequences that repeat more than k times."""
    tokens = re.findall(r"\S+|\s+", str(text))
    non_ws_tokens = []
    idx_map = []
    for idx, tok in enumerate(tokens):
        if not tok.isspace():
            idx_map.append(idx)
            non_ws_tokens.append(tok)

    n = len(non_ws_tokens)
    i = 0
    token_idx = 0
    out_tokens = []

    while i < n:
        replaced = False
        max_len = (n - i) // k
        for length in range(1, max_len + 1):
            seq = non_ws_tokens[i:i + length]
            count = 1
            while (
                i + (count + 1) * length <= n
                and non_ws_tokens[i + count * length:i + (count + 1) * length] == seq
            ):
                count += 1

            if count > k:
                start_token_idx = idx_map[i]
                end_token_idx = idx_map[i + length * k - 1] + 1
                out_tokens.extend(tokens[start_token_idx:end_token_idx])
                if tag:
                    out_tokens.append(" " + tag + " ")
                i += count * length
                token_idx = idx_map[i] if i < len(idx_map) else len(tokens)
                replaced = True
                break

        if not replaced:
            if token_idx < len(tokens):
                out_tokens.append(tokens[token_idx])
                token_idx += 1
                i += 1
                while token_idx < len(tokens) and tokens[token_idx].isspace():
                    out_tokens.append(tokens[token_idx])
                    token_idx += 1

    if token_idx < len(tokens):
        out_tokens.extend(tokens[token_idx:])

    return "".join(out_tokens)


def most_common_words(df: pd.DataFrame, proportion: float = 0.1, verbose: bool = False) -> List[str]:
    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    means = tfidf_matrix.mean(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    tfidf_scores = sorted(zip(vocab, means), key=lambda item: item[1], reverse=True)
    n_show = max(1, int(len(tfidf_scores) * proportion))
    top_words = tfidf_scores[:n_show]

    if verbose:
        for word, score in top_words:
            print(f"{word}: {score:.4f}")

    return [word for word, _ in top_words]
