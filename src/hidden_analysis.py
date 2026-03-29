from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PyTorch is required for hidden-state analysis. Install `torch` first."
    ) from exc


def forward_pass(model, enc, device):
    """Run a forward pass and return hidden states plus attentions."""
    model_inputs = {
        key: value.to(device)
        for key, value in enc.items()
        if key not in {"offset_mapping", "special_tokens_mask"}
    }
    with torch.no_grad():
        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            output_attentions=True,
        )
    return outputs.hidden_states, outputs.attentions


def pool_span_hidden(hidden_state, span: tuple[int, int]):
    """Average all token vectors inside a target span."""
    start, end = span
    return hidden_state[start:end].mean(dim=0)


def layerwise_word_representations(hidden_states, span: tuple[int, int]):
    """Extract one pooled vector per hidden-state layer for the target word span."""
    reps = []
    for hidden_state in hidden_states:
        reps.append(pool_span_hidden(hidden_state[0], span))
    return reps


def mean_pool_sentence(hidden_state, attention_mask, special_tokens_mask=None):
    """
    Mean-pool a sentence representation.

    By default, special tokens are excluded from the average because they can
    dominate sentence-level comparisons in short templates.
    """
    mask = attention_mask.clone().float()
    if special_tokens_mask is not None:
        mask = mask * (1.0 - special_tokens_mask.float())
    if mask.sum().item() == 0:
        mask = attention_mask.float()
    mask = mask.unsqueeze(-1)
    masked = hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def layerwise_sentence_representations(hidden_states, attention_mask, special_tokens_mask=None):
    """Extract one pooled sentence vector per hidden-state layer."""
    reps = []
    for hidden_state in hidden_states:
        reps.append(mean_pool_sentence(hidden_state, attention_mask, special_tokens_mask)[0])
    return reps


def cosine_similarity_list(reps1, reps2):
    """Compute cosine similarities across aligned layer representations."""
    return [
        F.cosine_similarity(rep1.unsqueeze(0), rep2.unsqueeze(0)).item()
        for rep1, rep2 in zip(reps1, reps2)
    ]
