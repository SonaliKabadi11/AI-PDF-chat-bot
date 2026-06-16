"""
PDF Semantic Search Pipeline — stateless, session-scoped functions.
Each uploaded PDF gets its own session_id so models/databases don't collide.
"""

from __future__ import annotations

import logging
import math
import pickle
import re
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
import numpy as np
from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

from config import EmbeddingConfig, ChunkingConfig, LLMConfig, PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. PDF Reading
# ---------------------------------------------------------------------------

def read_pdf(filepath: Path) -> List[str]:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    pages: List[str] = []
    with open(filepath, "rb") as fh:
        reader = PdfReader(fh)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(text.lower())
            else:
                logger.warning("Page %d yielded no text — skipping.", page_num + 1)

    if not pages:
        raise ValueError(
            f"No extractable text found in '{filepath}'. "
            "The PDF may be scanned or image-based."
        )

    logger.info("Read %d pages from '%s'.", len(pages), filepath.name)
    return pages


# ---------------------------------------------------------------------------
# 2. Text Chunking
# ---------------------------------------------------------------------------

def get_chunks(text: str, cfg: ChunkingConfig) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    chunks = splitter.split_text(text)
    logger.info("Created %d chunks.", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# 3. Tokenisation
# ---------------------------------------------------------------------------

def build_tokenizer(sentences: List[str]) -> Tokenizer:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    logger.info("Vocabulary size: %d tokens.", len(tokenizer.word_index) + 1)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(tokenizer, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(path: Path) -> Tokenizer:
    with open(Path(path), "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# 4. Training-data Generation
# ---------------------------------------------------------------------------

def generate_training_data(
    sentences: List[str],
    tokenizer: Tokenizer,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_list: List[int] = []
    y_list: List[int] = []

    for sentence in sentences:
        sequence = tokenizer.texts_to_sequences([sentence])[0]
        seq_len = len(sequence)
        for i, target_word in enumerate(sequence):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    x_list.append(target_word)
                    y_list.append(sequence[j])

    logger.info("Generated %d training pairs.", len(x_list))
    return np.array(x_list, dtype=np.int32), np.array(y_list, dtype=np.int32)


# ---------------------------------------------------------------------------
# 5. Embedding Model
# ---------------------------------------------------------------------------

def build_and_train_model(
    vocab_size: int,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    cfg: EmbeddingConfig,
) -> Sequential:
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=cfg.embedding_dim,
                input_length=1,
                name=cfg.layer_name,
            ),
            Flatten(),
            Dense(vocab_size, activation="softmax"),
        ],
        name="skip_gram_model",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("Training: %d pairs | vocab=%d | epochs=%d", len(X_train), vocab_size, cfg.epochs)
    model.fit(X_train, Y_train, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0)
    return model


def get_embedding_weights(model: Sequential, layer_name: str) -> np.ndarray:
    return model.get_layer(layer_name).get_weights()[0]


def save_model(model: Sequential, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_model_from_path(path: Path) -> Sequential:
    from tensorflow.keras.models import load_model
    return load_model(Path(path))


# ---------------------------------------------------------------------------
# 6. Tiny Transformer LLM
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="AIPDFChat")
class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        positions = self.pos_emb(positions)
        return self.token_emb(x) + positions

    def build(self, input_shape):
        self.token_emb.build(input_shape)
        self.pos_emb.build((self.maxlen,))
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "maxlen": self.maxlen,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="AIPDFChat")
class TransformerBlock(Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.att = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, embed_dim // num_heads),
        )
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)],
            name="feed_forward",
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        mask = tf.cast(i >= j, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        return tf.tile(mask, [batch_size, 1, 1])

    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(
            batch_size, seq_len, seq_len, tf.bool
        )
        attn_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def build(self, input_shape):
        self.att.build(input_shape, input_shape)
        self.ffn.build(input_shape)
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        self.dropout1.build(input_shape)
        self.dropout2.build(input_shape)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config


def build_transformer_lm(vocab_size: int, cfg: LLMConfig) -> Model:
    inputs = Input(shape=(cfg.sequence_length,), dtype=tf.int32)
    x = TokenAndPositionEmbedding(cfg.sequence_length, vocab_size, cfg.embedding_dim)(inputs)
    x = TransformerBlock(cfg.embedding_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout)(x)
    outputs = Dense(vocab_size, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs, name="tiny_pdf_transformer_lm")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def generate_lm_training_data(
    texts: List[str],
    tokenizer: Tokenizer,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    token_ids = tokenizer.texts_to_sequences([" ".join(texts)])[0]
    if len(token_ids) <= sequence_length:
        return np.empty((0, sequence_length), dtype=np.int32), np.empty((0, sequence_length), dtype=np.int32)

    inputs, targets = [], []
    for i in range(0, len(token_ids) - sequence_length):
        window = token_ids[i : i + sequence_length + 1]
        inputs.append(window[:-1])
        targets.append(window[1:])
    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)


def build_and_train_llm(
    texts: List[str],
    tokenizer: Tokenizer,
    vocab_size: int,
    cfg: LLMConfig,
) -> Tuple[Model | None, dict]:
    X_train, Y_train = generate_lm_training_data(texts, tokenizer, cfg.sequence_length)
    if len(X_train) == 0:
        logger.warning("Skipping LLM training: not enough tokens for sequence_length=%d.", cfg.sequence_length)
        return None, {
            "trained": False,
            "samples": 0,
            "loss": None,
            "accuracy": None,
            "perplexity": None,
        }

    model = build_transformer_lm(vocab_size, cfg)
    logger.info(
        "Training tiny transformer LLM: samples=%d | vocab=%d | epochs=%d",
        len(X_train),
        vocab_size,
        cfg.epochs,
    )
    history = model.fit(X_train, Y_train, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0)
    final_loss = float(history.history["loss"][-1])
    final_accuracy = float(history.history.get("accuracy", [0.0])[-1])
    return model, {
        "trained": True,
        "samples": int(len(X_train)),
        "loss": round(final_loss, 4),
        "accuracy": round(final_accuracy, 4),
        "perplexity": round(float(math.exp(min(final_loss, 20))), 4),
    }


def save_llm_model(model: Model, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_llm_model(path: Path) -> Model:
    from tensorflow.keras.models import load_model
    return load_model(
        Path(path),
        custom_objects={
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerBlock": TransformerBlock,
        },
        compile=False,
    )


def build_rag_prompt(query_text: str, retrieved_chunks: List[dict]) -> str:
    context = " ".join(item["text"] for item in retrieved_chunks)
    return (
        "question "
        + query_text.strip().lower()
        + " context "
        + context.strip().lower()
        + " answer"
    )


def _sample_next_token(probabilities: np.ndarray, temperature: float, top_k: int) -> int:
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(np.maximum(probabilities, 1e-9)) / max(temperature, 1e-6)
    if top_k > 0 and top_k < len(probabilities):
        top_indices = np.argpartition(probabilities, -top_k)[-top_k:]
        filtered = np.full_like(probabilities, -np.inf)
        filtered[top_indices] = probabilities[top_indices]
        probabilities = filtered
    exp_probs = np.exp(probabilities - np.max(probabilities))
    probabilities = exp_probs / np.sum(exp_probs)
    return int(np.random.choice(len(probabilities), p=probabilities))


def _apply_repetition_penalty(
    probabilities: np.ndarray,
    generated_ids: List[int],
    penalty: float,
) -> np.ndarray:
    if penalty <= 1.0 or not generated_ids:
        return probabilities
    adjusted = probabilities.copy()
    for token_id in set(generated_ids):
        if 0 <= token_id < len(adjusted):
            adjusted[token_id] /= penalty
    total = np.sum(adjusted)
    return adjusted / total if total > 0 else probabilities


def _count_repeated_ngrams(token_ids: List[int], ngram_size: int) -> int:
    if ngram_size <= 1 or len(token_ids) < ngram_size:
        return 0
    seen = set()
    repeats = 0
    for i in range(len(token_ids) - ngram_size + 1):
        ngram = tuple(token_ids[i : i + ngram_size])
        if ngram in seen:
            repeats += 1
        else:
            seen.add(ngram)
    return repeats


def repetition_score(text: str) -> float:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    bigrams = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
    repeated_bigrams = len(bigrams) - len(set(bigrams))
    repetition_penalty = repeated_bigrams / max(len(bigrams), 1)
    return round(float(max(0.0, unique_ratio - repetition_penalty)), 4)


def has_repetitive_generation(text: str, ngram_size: int = 6, max_repeats: int = 1) -> bool:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < ngram_size * 2:
        return False
    ngrams = []
    for i in range(len(words) - ngram_size + 1):
        ngrams.append(tuple(words[i : i + ngram_size]))
    repeated = len(ngrams) - len(set(ngrams))
    return repeated > max_repeats


def _clean_extractive_answer(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" .")
    cleaned = re.sub(r"ofn\s*=", "of n =", cleaned, flags=re.I)
    cleaned = cleaned.lstrip("•-* ")
    cleaned = re.sub(
        r"^(to the best of our knowledge,\s*however,\s*)",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"^(similarly,\s*|in addition,\s*|however,\s*)", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^decoder:\s*the decoder\b", "the decoder", cleaned, flags=re.I)
    cleaned = re.sub(r"^decoder:\s*", "the decoder ", cleaned, flags=re.I)
    cleaned = re.sub(r"^\d+\s+", "", cleaned)
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned + "." if cleaned and cleaned[-1] not in ".!?" else cleaned


def _query_concept(query_text: str) -> str:
    words = list(_content_words(query_text))
    if not words:
        return query_text.strip().lower().strip(" ?.")
    ordered_words = [
        word
        for word in re.findall(r"[a-z0-9]+", query_text.lower())
        if word in words
    ]
    return " ".join(ordered_words).strip()


def _split_candidate_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    parts = re.split(r"(?<=[.!?])\s+|(?=\s*[•-]\s+)", normalized)
    candidates: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # PDF extraction often removes periods. Split around strong clause starts
        # while keeping enough context for standalone evidence.
        subparts = re.split(
            r"(?=\b(?:decoder:|encoder:|in addition|similarly|each decoder)\b)",
            part,
            flags=re.I,
        )
        for subpart in subparts:
            subpart = subpart.strip()
            if len(subpart) > 4:
                candidates.append(subpart)
    return candidates or [normalized]


def _is_weak_fragment(sentence: str) -> bool:
    words = re.findall(r"[a-z0-9]+", sentence.lower())
    if not words:
        return True
    weak_endings = {"to", "of", "and", "the", "in"}
    transition_starts = {
        "also",
        "encoder",
        "however",
        "in",
        "similarly",
        "therefore",
        "thus",
        "while",
    }
    lowered = sentence.strip().lower()
    return (
        words[-1] in weak_endings
        or words[0] in transition_starts
        or lowered.startswith(("•", "-", "*"))
        or len(words) < 5
    )


def _definition_sentence_score(query_text: str, sentence: str, distance: float) -> float:
    query_terms = _content_words(query_text)
    sentence_terms = _content_words(sentence)
    lowered = f" {sentence.lower()} "

    overlap = len(query_terms & sentence_terms)
    coverage = _safe_ratio(overlap, len(query_terms))
    score = (overlap * 2.0) + (coverage * 3.0) - distance

    definition_patterns = (
        r"\bis\s+a\b",
        r"\bis\s+an\b",
        r"\bis\s+the\b",
        r"\bis\s+also\s+composed\b",
        r"\bare\s+a\b",
        r"\bare\s+the\b",
        r"\bcomposed\s+of\b",
        r"\brefers\s+to\b",
        r"\bdefined\s+as\b",
        r"\bbased\s+entirely\s+on\b",
        r"\brelying\s+entirely\s+on\b",
        r"\breplacing\b",
    )
    if any(re.search(pattern, lowered) for pattern in definition_patterns):
        score += 3.0

    useful_terms = ("model", "architecture", "attention", "self-attention", "mechanism")
    score += sum(0.35 for term in useful_terms if term in lowered)

    if "decoder" in query_terms:
        decoder_terms = (
            "decoder stack",
            "decoder is also composed",
            "decoder composed",
            "decoder layer",
            "decoder inserts",
            "encoder stack",
            "multi-head attention over the output",
            "self-attention layers in the decoder",
            "feed-forward network",
            "up to and including that position",
        )
        score += sum(1.2 for term in decoder_terms if term in lowered)

    weak_terms = (
        "bleu",
        "configuration",
        "dropout",
        "gpu",
        "gpus",
        "score",
        "table",
        "trained for",
        "training took",
    )
    score -= sum(1.5 for term in weak_terms if term in lowered)
    if "decoder" in query_terms and "encoder and decoder stacks encoder" in lowered:
        score -= 4.0

    word_count = len(re.findall(r"[a-z0-9]+", sentence.lower()))
    if word_count < 6:
        score -= 2.0
    elif word_count > 45:
        score -= 0.75
    if _is_weak_fragment(sentence):
        score -= 2.5
    if sentence.strip()[-1:] in ".!?":
        score += 0.5
    if query_terms & sentence_terms:
        score += 0.75
    return score


def _evidence_aspects(sentence: str) -> set[str]:
    lowered = sentence.lower()
    aspects = set()
    if "stack" in lowered or "composed" in lowered or "layer" in lowered:
        aspects.add("structure")
    if "encoder stack" in lowered or "output of the encoder" in lowered:
        aspects.add("encoder_decoder_attention")
    if "self-attention" in lowered or "up to and including" in lowered:
        aspects.add("masked_self_attention")
    if "feed-forward" in lowered or "feed forward" in lowered:
        aspects.add("feed_forward")
    return aspects


def build_definition_answer(query_text: str, retrieved_chunks: List[dict], max_sentences: int = 4) -> str:
    concept = _query_concept(query_text)
    if not retrieved_chunks:
        return "No relevant context was retrieved from the PDF."

    scored_candidates: List[Tuple[float, str]] = []
    for chunk in retrieved_chunks:
        distance = float(chunk.get("distance", 0.0))
        for sentence in _split_candidate_sentences(chunk["text"]):
            cleaned = _clean_extractive_answer(sentence)
            if not cleaned:
                continue
            score = _definition_sentence_score(query_text, cleaned, distance)
            scored_candidates.append((score, cleaned))

    if not scored_candidates:
        return _clean_extractive_answer(retrieved_chunks[0]["text"])

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    selected: List[str] = []
    selected_terms: set[str] = set()
    selected_aspects: set[str] = set()
    concept_terms = _content_words(concept)

    for score, sentence in scored_candidates:
        sentence_terms = _content_words(sentence)
        if concept_terms and not (concept_terms & sentence_terms):
            continue
        if any(sentence.lower() == existing.lower() for existing in selected):
            continue

        # Prefer evidence that adds something new instead of repeating the same clause.
        novelty = sentence_terms - selected_terms
        if selected and len(novelty) < 3:
            continue
        aspects = _evidence_aspects(sentence)
        if selected and aspects and aspects.issubset(selected_aspects):
            continue

        selected.append(sentence)
        selected_terms.update(sentence_terms)
        selected_aspects.update(aspects)
        if len(selected) >= max_sentences:
            break

    for _score, sentence in scored_candidates:
        if len(selected) >= max_sentences:
            break
        if any(sentence.lower() == existing.lower() for existing in selected):
            continue
        aspects = _evidence_aspects(sentence)
        if aspects and not aspects.issubset(selected_aspects):
            selected.append(sentence)
            selected_aspects.update(aspects)

    if not selected:
        selected = [scored_candidates[0][1]]

    first = selected[0]
    concept_label = concept or "the requested concept"
    if first.lower().startswith(concept_label.lower()):
        intro = f"Based on the PDF, {first}"
    else:
        intro = f"Based on the PDF, {concept_label} is described as follows: {first}"

    details = selected[1:]
    if details:
        return f"{intro} It includes " + " ".join(details)
    return intro


def extractive_fallback_answer(query_text: str, retrieved_chunks: List[dict]) -> str:
    if is_definition_query(query_text):
        return build_definition_answer(query_text, retrieved_chunks)

    query_words = _content_words(query_text)
    if not retrieved_chunks:
        return "No relevant context was retrieved from the PDF."

    best_sentence = retrieved_chunks[0]["text"]
    best_score = -1.0
    for chunk in retrieved_chunks:
        sentences = _split_candidate_sentences(chunk["text"])
        for sentence in sentences or [chunk["text"]]:
            sentence_words = _content_words(sentence)
            overlap = len(query_words & sentence_words)
            coverage = _safe_ratio(overlap, len(query_words))
            score = (overlap * 2.0) + coverage - float(chunk.get("distance", 0.0))
            if _is_weak_fragment(sentence):
                score -= 1.0
            if score > best_score:
                best_score = score
                best_sentence = sentence
    return _clean_extractive_answer(best_sentence)


def generate_answer(
    query_text: str,
    retrieved_chunks: List[dict],
    llm_model: Model,
    tokenizer: Tokenizer,
    cfg: LLMConfig,
) -> Tuple[str, dict]:
    if is_definition_query(query_text):
        return (
            extractive_fallback_answer(query_text, retrieved_chunks),
            {
                "answer_source": "extractive",
                "fallback_used": True,
                "fallback_reason": "definition_query",
                "generated_token_count": 0,
                "query_type": "definition",
            },
        )

    prompt = build_rag_prompt(query_text, retrieved_chunks)
    generated_ids = tokenizer.texts_to_sequences([prompt])[0]
    if not generated_ids:
        return (
            extractive_fallback_answer(query_text, retrieved_chunks),
            {
                "answer_source": "extractive",
                "fallback_used": True,
                "fallback_reason": "query_words_not_in_vocabulary",
                "generated_token_count": 0,
                "query_type": "general",
            },
        )

    prompt_len = len(generated_ids)
    answer_ids: List[int] = []
    for _ in range(cfg.max_new_tokens):
        model_input = generated_ids[-cfg.sequence_length:]
        if len(model_input) < cfg.sequence_length:
            model_input = [0] * (cfg.sequence_length - len(model_input)) + model_input
        prediction = llm_model.predict(np.array([model_input], dtype=np.int32), verbose=0)[0]
        probabilities = _apply_repetition_penalty(
            prediction[-1],
            answer_ids,
            cfg.repetition_penalty,
        )
        next_id = _sample_next_token(probabilities, cfg.temperature, cfg.top_k)
        if next_id == 0:
            break
        generated_ids.append(next_id)
        answer_ids.append(next_id)
        if _count_repeated_ngrams(
            answer_ids,
            cfg.repeat_ngram_size,
        ) > cfg.max_repeated_ngrams:
            break

    answer_ids = generated_ids[prompt_len:]
    answer = tokenizer.sequences_to_texts([answer_ids])[0].strip()
    metadata = {
        "answer_source": "transformer",
        "fallback_used": False,
        "fallback_reason": None,
        "generated_token_count": len(answer_ids),
        "query_type": "general",
    }
    if not answer:
        metadata["fallback_used"] = True
        metadata["fallback_reason"] = "empty_generation"
        metadata["answer_source"] = "extractive"
        return extractive_fallback_answer(query_text, retrieved_chunks), metadata
    if has_repetitive_generation(
        answer,
        cfg.repeat_ngram_size,
        cfg.max_repeated_ngrams,
    ):
        metadata["fallback_used"] = True
        metadata["fallback_reason"] = "repetitive_generation"
        metadata["answer_source"] = "extractive"
        return extractive_fallback_answer(query_text, retrieved_chunks), metadata
    return answer, metadata


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


_QUESTION_STOPWORDS = {
    "a",
    "about",
    "an",
    "are",
    "define",
    "does",
    "explain",
    "for",
    "give",
    "how",
    "in",
    "is",
    "me",
    "of",
    "tell",
    "the",
    "this",
    "to",
    "what",
    "whats",
    "what's",
    "why",
}


def _content_words(text: str) -> set[str]:
    return _word_set(text) - _QUESTION_STOPWORDS


def is_definition_query(query_text: str) -> bool:
    query = query_text.strip().lower()
    definition_starts = (
        "what is ",
        "what are ",
        "define ",
        "explain ",
        "tell me about ",
    )
    return query.startswith(definition_starts) or query.startswith("what's ")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def evaluate_answer_metrics(
    query_text: str,
    answer: str,
    retrieved_chunks: List[dict],
    tokenizer: Tokenizer,
    vocab_size: int,
    weights: np.ndarray,
    embedding_dim: int,
) -> dict:
    context = " ".join(item["text"] for item in retrieved_chunks)
    answer_words = _word_set(answer)
    query_words = _content_words(query_text)
    context_words = _word_set(context)

    grounded_words = answer_words & context_words
    query_words_in_answer = query_words & answer_words

    context_grounding = _safe_ratio(len(grounded_words), len(answer_words))
    query_coverage = _safe_ratio(len(query_words_in_answer), len(query_words))

    answer_vector = text_to_vector(answer, tokenizer, vocab_size, weights, embedding_dim)
    context_vector = text_to_vector(context, tokenizer, vocab_size, weights, embedding_dim)
    query_vector = text_to_vector(query_text, tokenizer, vocab_size, weights, embedding_dim)

    answer_context_similarity = max(0.0, _cosine_similarity(answer_vector, context_vector))
    answer_query_similarity = max(0.0, _cosine_similarity(answer_vector, query_vector))

    distances = [float(item["distance"]) for item in retrieved_chunks if "distance" in item]
    avg_distance = float(np.mean(distances)) if distances else 0.0
    best_distance = float(np.min(distances)) if distances else 0.0
    retrieval_confidence = 1.0 / (1.0 + max(avg_distance, 0.0))

    estimated_accuracy = np.mean(
        [
            context_grounding,
            query_coverage,
            answer_context_similarity,
            retrieval_confidence,
        ]
    )
    fluency_score = repetition_score(answer)

    return {
        "estimated_accuracy": round(float(estimated_accuracy), 4),
        "context_grounding": round(float(context_grounding), 4),
        "query_coverage": round(float(query_coverage), 4),
        "answer_context_similarity": round(float(answer_context_similarity), 4),
        "answer_query_similarity": round(float(answer_query_similarity), 4),
        "retrieval_confidence": round(float(retrieval_confidence), 4),
        "avg_retrieval_distance": round(avg_distance, 4),
        "best_retrieval_distance": round(best_distance, 4),
        "answer_word_count": len(answer_words),
        "grounded_word_count": len(grounded_words),
        "fluency_score": fluency_score,
        "repetition_detected": has_repetitive_generation(answer),
    }


def should_use_extractive_answer(metrics: dict, cfg: LLMConfig) -> Tuple[bool, str | None]:
    if metrics.get("repetition_detected"):
        return True, "repetition_detected"
    if metrics.get("context_grounding", 0.0) < cfg.min_grounding_score:
        return True, "low_grounding"
    if metrics.get("query_coverage", 0.0) < cfg.min_query_coverage:
        return True, "low_query_coverage"
    if metrics.get("fluency_score", 0.0) < cfg.min_fluency_score:
        return True, "low_fluency"
    if metrics.get("estimated_accuracy", 0.0) < cfg.min_estimated_accuracy:
        return True, "low_estimated_accuracy"
    return False, None


# ---------------------------------------------------------------------------
# 7. Vector Utilities
# ---------------------------------------------------------------------------

def get_word_vector(
    word: str,
    tokenizer: Tokenizer,
    vocab_size: int,
    weights: np.ndarray,
    embedding_dim: int,
) -> np.ndarray:
    word_idx = tokenizer.word_index.get(word.lower())
    if word_idx is not None and word_idx < vocab_size:
        return weights[word_idx]
    return np.zeros(embedding_dim, dtype=np.float32)


def text_to_vector(
    text: str,
    tokenizer: Tokenizer,
    vocab_size: int,
    weights: np.ndarray,
    embedding_dim: int,
) -> np.ndarray | None:
    word_vectors = [
        get_word_vector(w, tokenizer, vocab_size, weights, embedding_dim)
        for w in text.lower().split()
        if tokenizer.word_index.get(w.lower(), 0) > 0
    ]
    if not word_vectors:
        return None
    return np.mean(word_vectors, axis=0).astype(np.float32)


def chunks_to_vectors(
    chunks: List[str],
    tokenizer: Tokenizer,
    vocab_size: int,
    weights: np.ndarray,
    embedding_dim: int,
) -> List[np.ndarray]:
    vectors = []
    for chunk in chunks:
        vec = text_to_vector(chunk, tokenizer, vocab_size, weights, embedding_dim)
        vectors.append(vec if vec is not None else np.zeros(embedding_dim, dtype=np.float32))
    return vectors


# ---------------------------------------------------------------------------
# 8. ChromaDB Vector Store
# ---------------------------------------------------------------------------

def _chroma_client(db_path: Path) -> PersistentClient:
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    return PersistentClient(path=str(db_path))


def store_chunk_vectors(
    chunks: List[str],
    chunk_vectors: List[np.ndarray],
    db_path: Path,
    collection_name: str,
) -> None:
    client = _chroma_client(db_path)
    collection = client.get_or_create_collection(name=collection_name)
    collection.upsert(
        documents=chunks,
        embeddings=[v.tolist() for v in chunk_vectors],
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    logger.info("Stored %d chunks → '%s'.", len(chunks), collection_name)


def store_sentence_vectors(
    full_text: str,
    tokenizer: Tokenizer,
    vocab_size: int,
    weights: np.ndarray,
    embedding_dim: int,
    db_path: Path,
    collection_name: str,
    batch_size: int = 500,
) -> None:
    raw_sentences = [s.strip() for s in full_text.split(".") if len(s.strip()) > 5]
    sentences_to_insert, sentence_vectors, sentence_ids = [], [], []

    for idx, sentence in enumerate(raw_sentences):
        vec = text_to_vector(sentence, tokenizer, vocab_size, weights, embedding_dim)
        if vec is None:
            continue
        sentences_to_insert.append(sentence)
        sentence_vectors.append(vec.tolist())
        sentence_ids.append(f"sent_{idx}")

    client = _chroma_client(db_path)
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
    for i in range(0, len(sentences_to_insert), batch_size):
        end = i + batch_size
        collection.upsert(
            ids=sentence_ids[i:end],
            embeddings=sentence_vectors[i:end],
            documents=sentences_to_insert[i:end],
        )
    logger.info("Indexed %d sentences → '%s'.", len(sentences_to_insert), collection_name)


# ---------------------------------------------------------------------------
# 9. Semantic Search
# ---------------------------------------------------------------------------

def search_pdf(
    query_text: str,
    tokenizer: Tokenizer,
    vocab_size: int,
    weights: np.ndarray,
    embedding_dim: int,
    db_path: Path,
    collection_name: str,
    top_n: int = 5,
) -> List[dict]:
    query_vector = text_to_vector(query_text, tokenizer, vocab_size, weights, embedding_dim)
    if query_vector is None:
        raise ValueError(f"No query words recognised by model for: '{query_text}'")

    client = _chroma_client(db_path)
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
    results = collection.query(
        query_embeddings=[query_vector.tolist()], n_results=top_n
    )
    return [
        {"text": doc, "distance": round(float(dist), 4)}
        for doc, dist in zip(results["documents"][0], results["distances"][0])
    ]


# ---------------------------------------------------------------------------
# 10. Full ingestion pipeline (called after upload)
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: Path, session_id: str, cfg: PipelineConfig) -> dict:
    """
    Run the full ingestion pipeline for a single uploaded PDF.
    Returns a summary dict with session_id and stats.
    """
    pages = read_pdf(pdf_path)
    full_text = "\n".join(pages)
    chunks = get_chunks(full_text, cfg.chunking)

    tokenizer = build_tokenizer(chunks)
    vocab_size = len(tokenizer.word_index) + 1

    tokenizer_path = cfg.tokenizer_save_dir / f"{session_id}.pickle"
    save_tokenizer(tokenizer, tokenizer_path)

    X_train, Y_train = generate_training_data(chunks, tokenizer, cfg.embedding.window_size)
    model = build_and_train_model(vocab_size, X_train, Y_train, cfg.embedding)

    model_path = cfg.model_save_dir / f"{session_id}.keras"
    save_model(model, model_path)

    weights = get_embedding_weights(model, cfg.embedding.layer_name)

    llm_model, llm_metrics = build_and_train_llm(chunks, tokenizer, vocab_size, cfg.llm)
    if llm_model is not None:
        llm_model_path = cfg.llm_save_dir / f"{session_id}.keras"
        save_llm_model(llm_model, llm_model_path)

    chunk_vectors = chunks_to_vectors(chunks, tokenizer, vocab_size, weights, cfg.embedding.embedding_dim)
    store_chunk_vectors(
        chunks, chunk_vectors,
        cfg.vector_db_base / session_id,
        cfg.chunk_collection_name,
    )
    store_sentence_vectors(
        full_text, tokenizer, vocab_size, weights, cfg.embedding.embedding_dim,
        cfg.chroma_db_base / session_id,
        cfg.sentence_collection_name,
        batch_size=cfg.chroma_batch_size,
    )

    return {
        "session_id": session_id,
        "pages": len(pages),
        "chunks": len(chunks),
        "vocab_size": vocab_size,
        "training_pairs": len(X_train),
        "llm_trained": llm_model is not None,
        "llm_metrics": llm_metrics,
    }
