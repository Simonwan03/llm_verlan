from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PyTorch is required for attention analysis. Install `torch` first."
    ) from exc


def span_indices(span: tuple[int, int]) -> list[int]:
    start, end = span
    return list(range(start, end))


def compute_attention_breakdown(attentions, span: tuple[int, int]):
    """
    Aggregate outgoing attention from the target span into three buckets:
    self, intra-word, and context.
    """
    self_per_layer = []
    intra_per_layer = []
    context_per_layer = []

    target_ids = span_indices(span)

    for layer_attn in attentions:
        layer_matrix = layer_attn[0].detach().cpu()
        _, seq_len, _ = layer_matrix.shape
        target_mask = torch.zeros(seq_len, dtype=torch.bool)
        target_mask[target_ids] = True

        self_vals = []
        intra_vals = []
        context_vals = []

        for head_matrix in layer_matrix:
            target_rows = head_matrix[target_ids]
            total = target_rows.sum().item()
            if total == 0:
                self_vals.append(0.0)
                intra_vals.append(0.0)
                context_vals.append(0.0)
                continue

            target_to_target = target_rows[:, target_mask]
            self_attn = target_to_target.diag().sum().item()
            intra_attn = target_to_target.sum().item() - self_attn
            context_attn = target_rows[:, ~target_mask].sum().item()

            self_vals.append(self_attn / total)
            intra_vals.append(intra_attn / total)
            context_vals.append(context_attn / total)

        self_per_layer.append(sum(self_vals) / len(self_vals))
        intra_per_layer.append(sum(intra_vals) / len(intra_vals))
        context_per_layer.append(sum(context_vals) / len(context_vals))

    return {
        "self": self_per_layer,
        "intra": intra_per_layer,
        "context": context_per_layer,
    }


def compute_incoming_attention(attentions, span: tuple[int, int], exclude_target_sources: bool = True):
    """
    Measure how much attention the rest of the sentence sends into the target span.

    When `exclude_target_sources` is True, rows belonging to the target span are
    omitted so the metric captures incoming attention from context only.
    """
    incoming_per_layer = []
    target_ids = span_indices(span)

    for layer_attn in attentions:
        layer_matrix = layer_attn[0].detach().cpu()
        _, seq_len, _ = layer_matrix.shape
        target_mask = torch.zeros(seq_len, dtype=torch.bool)
        target_mask[target_ids] = True
        source_mask = ~target_mask if exclude_target_sources else torch.ones(seq_len, dtype=torch.bool)

        per_head_values = []
        for head_matrix in layer_matrix:
            source_rows = head_matrix[source_mask]
            total = source_rows.sum().item()
            if total == 0:
                per_head_values.append(0.0)
                continue
            incoming = source_rows[:, target_mask].sum().item()
            per_head_values.append(incoming / total)

        incoming_per_layer.append(sum(per_head_values) / len(per_head_values))

    return incoming_per_layer
