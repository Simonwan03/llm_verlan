#!/usr/bin/env python3
"""Analyze standard vs informal French sentence pairs with French language models.

This script:
1. Loads a CSV file into a pandas DataFrame.
2. Tokenizes standard and informal sentences.
3. Computes masked-LM pseudo-log-likelihood (PLL) or causal-LM NLL/perplexity.
4. Extracts mean-pooled last-layer hidden-state representations.
5. Computes cosine similarity between the two sentence representations.
6. Writes the enriched DataFrame back to disk.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
)

MODEL_NAME = "camembert-base"
MODEL_ALIASES = {
    "camembert": "camembert-base",
    "camembert-base": "camembert-base",
    "camembert_v2": "almanach/camembertv2-base",
    "camembertv2": "almanach/camembertv2-base",
    "clair-7b-0.1": "OpenLLM-France/Claire-7B-0.1",
    "claire-7b-0.1": "OpenLLM-France/Claire-7B-0.1",
    "claire_7b_0_1": "OpenLLM-France/Claire-7B-0.1",
}
REQUIRED_COLUMNS = [
    "id",
    "phenomenon",
    "standard_sentence",
    "informal_sentence",
    "meaning_same",
    "notes",
]


def normalize_text(value: Any) -> str:
    """Return a clean string; treat NaN/None/blank as an empty sentence."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def validate_dataframe(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )


def resolve_model_name(model_name: str) -> str:
    """Map friendly aliases to Hugging Face model identifiers."""
    return MODEL_ALIASES.get(model_name, model_name)


def infer_model_family(config: AutoConfig) -> str:
    """Return the scoring family supported by the loaded checkpoint."""
    model_type = config.model_type
    if model_type in MODEL_FOR_MASKED_LM_MAPPING_NAMES:
        return "masked_lm"
    if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return "causal_lm"

    raise ValueError(
        f"Unsupported model type '{model_type}'. "
        "This script supports masked-LM and causal-LM checkpoints."
    )


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """Load the tokenizer and model on GPU if available."""
    resolved_model_name = resolve_model_name(model_name)
    config = AutoConfig.from_pretrained(resolved_model_name)
    model_family = infer_model_family(config)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, use_fast=True)

    if model_family == "masked_lm":
        model = AutoModelForMaskedLM.from_pretrained(
            resolved_model_name,
            output_hidden_states=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            output_hidden_states=True,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device, model_family


@torch.inference_mode()
def tokenize_sentence(tokenizer, text: str) -> list[str]:
    """Return the subword tokens without adding special tokens."""
    if not text:
        return []
    return tokenizer.tokenize(text)


@torch.inference_mode()
def compute_pll_score(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 256,
    pll_batch_size: int = 32,
) -> float:
    """Compute pseudo-log-likelihood, normalized by scored token count.

    The function masks each non-special token once, scores the original token,
    sums the log-probabilities, and divides by the number of scored tokens.
    """
    if not text:
        return math.nan

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    special_tokens_mask = encoded["special_tokens_mask"][0].to(device).bool()

    candidate_positions = (
        attention_mask[0].bool() & ~special_tokens_mask
    ).nonzero(as_tuple=False).flatten()

    if candidate_positions.numel() == 0:
        return math.nan

    total_log_probability = 0.0

    for start in range(0, candidate_positions.numel(), pll_batch_size):
        chunk_positions = candidate_positions[start : start + pll_batch_size]
        batch_size = chunk_positions.numel()

        batch_input_ids = input_ids.repeat(batch_size, 1)
        batch_attention_mask = attention_mask.repeat(batch_size, 1)
        row_indices = torch.arange(batch_size, device=device)
        original_token_ids = batch_input_ids[row_indices, chunk_positions]

        batch_input_ids[row_indices, chunk_positions] = tokenizer.mask_token_id

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
        )
        masked_logits = outputs.logits[row_indices, chunk_positions, :]
        log_probabilities = F.log_softmax(masked_logits, dim=-1)
        selected_log_probabilities = log_probabilities.gather(
            1,
            original_token_ids.unsqueeze(1),
        ).squeeze(1)

        total_log_probability += selected_log_probabilities.sum().item()

    return total_log_probability / candidate_positions.numel()


@torch.inference_mode()
def compute_causal_lm_metrics(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 256,
) -> dict[str, float]:
    """Compute average token NLL and perplexity for a causal LM."""
    if not text:
        return {"nll_score": math.nan, "perplexity": math.nan}

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    if int(attention_mask[0].sum().item()) < 2:
        return {"nll_score": math.nan, "perplexity": math.nan}

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    average_nll = float(outputs.loss.item())
    perplexity = float(math.exp(average_nll))
    return {"nll_score": average_nll, "perplexity": perplexity}


@torch.inference_mode()
def get_mean_pooled_representation(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 256,
) -> torch.Tensor | None:
    """Return the mean-pooled last hidden state, ignoring padding tokens."""
    if not text:
        return None

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    last_hidden_state = outputs.hidden_states[-1]

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed_hidden_states = (last_hidden_state * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    pooled_representation = summed_hidden_states / token_counts

    return pooled_representation.squeeze(0).cpu()


def cosine_similarity(
    first_representation: torch.Tensor | None,
    second_representation: torch.Tensor | None,
) -> float:
    """Compute cosine similarity between two pooled sentence embeddings."""
    if first_representation is None or second_representation is None:
        return math.nan

    similarity = F.cosine_similarity(
        first_representation.unsqueeze(0),
        second_representation.unsqueeze(0),
        dim=1,
    )
    return float(similarity.item())


def analyze_unique_sentences(
    sentences: list[str],
    tokenizer,
    model,
    model_family: str,
    device: torch.device,
    max_length: int,
    pll_batch_size: int,
) -> dict[str, dict[str, Any]]:
    """Compute tokenization, scores, and representation once per unique sentence."""
    sentence_features: dict[str, dict[str, Any]] = {}

    for sentence in sentences:
        if sentence in sentence_features:
            continue

        tokens = tokenize_sentence(tokenizer, sentence)
        sentence_metrics = {
            "tokens": tokens,
            "token_count": len(tokens),
            "pll_score": math.nan,
            "nll_score": math.nan,
            "perplexity": math.nan,
            "representation": get_mean_pooled_representation(
                text=sentence,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
            ),
        }

        if model_family == "masked_lm":
            sentence_metrics["pll_score"] = compute_pll_score(
                text=sentence,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
                pll_batch_size=pll_batch_size,
            )
        else:
            sentence_metrics.update(
                compute_causal_lm_metrics(
                    text=sentence,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=max_length,
                )
            )

        sentence_features[sentence] = sentence_metrics

    return sentence_features


def enrich_dataframe(
    df: pd.DataFrame,
    tokenizer,
    model,
    model_family: str,
    device: torch.device,
    max_length: int = 256,
    pll_batch_size: int = 32,
) -> pd.DataFrame:
    """Add tokenization, sequence scores, and representation similarity columns."""
    validate_dataframe(df)

    enriched_df = df.copy()
    standard_texts = enriched_df["standard_sentence"].map(normalize_text)
    informal_texts = enriched_df["informal_sentence"].map(normalize_text)

    unique_sentences = pd.unique(
        pd.concat([standard_texts, informal_texts], ignore_index=True)
    ).tolist()

    features = analyze_unique_sentences(
        sentences=unique_sentences,
        tokenizer=tokenizer,
        model=model,
        model_family=model_family,
        device=device,
        max_length=max_length,
        pll_batch_size=pll_batch_size,
    )

    enriched_df["standard_tokens"] = standard_texts.map(lambda text: features[text]["tokens"])
    enriched_df["informal_tokens"] = informal_texts.map(lambda text: features[text]["tokens"])
    enriched_df["standard_token_count"] = standard_texts.map(
        lambda text: features[text]["token_count"]
    )
    enriched_df["informal_token_count"] = informal_texts.map(
        lambda text: features[text]["token_count"]
    )
    enriched_df["standard_pll_score"] = standard_texts.map(
        lambda text: features[text]["pll_score"]
    )
    enriched_df["informal_pll_score"] = informal_texts.map(
        lambda text: features[text]["pll_score"]
    )
    enriched_df["standard_nll_score"] = standard_texts.map(
        lambda text: features[text]["nll_score"]
    )
    enriched_df["informal_nll_score"] = informal_texts.map(
        lambda text: features[text]["nll_score"]
    )
    enriched_df["standard_perplexity"] = standard_texts.map(
        lambda text: features[text]["perplexity"]
    )
    enriched_df["informal_perplexity"] = informal_texts.map(
        lambda text: features[text]["perplexity"]
    )
    enriched_df["representation_cosine_similarity"] = [
        cosine_similarity(features[standard]["representation"], features[informal]["representation"])
        for standard, informal in zip(standard_texts, informal_texts)
    ]
    enriched_df["model_family"] = model_family

    return enriched_df


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze standard vs informal French sentence pairs with masked or causal LMs.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/paired_informal_french_dataset.csv"),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/paired_informal_french_dataset_camembert_analysis.csv"),
        help="Path to write the enriched CSV file.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=(
            "Hugging Face model name or alias to load. "
            "Built-in aliases: camembert, camembert-base, camembert_v2, "
            "clair-7b-0.1, claire-7b-0.1."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum tokenized sequence length used for inference.",
    )
    parser.add_argument(
        "--pll-batch-size",
        type=int,
        default=32,
        help="How many masked variants to score per forward pass.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    resolved_model_name = resolve_model_name(args.model_name)
    tokenizer, model, device, model_family = load_model_and_tokenizer(args.model_name)

    enriched_df = enrich_dataframe(
        df=df,
        tokenizer=tokenizer,
        model=model,
        model_family=model_family,
        device=device,
        max_length=args.max_length,
        pll_batch_size=args.pll_batch_size,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(args.output_csv, index=False)

    print(f"Loaded {len(df)} rows from: {args.input_csv}")
    print(f"Requested model: {args.model_name}")
    print(f"Resolved model: {resolved_model_name}")
    print(f"Model family: {model_family}")
    print(f"Using device: {device}")
    print(f"Wrote enriched DataFrame to: {args.output_csv}")


if __name__ == "__main__":
    main()
    # Example usage:
    # python scripts/camembert_analysis.py --model-name camembert/clair-7b-0.1 --output-csv data/paired_informal_french_dataset_camembert_analysis.csv