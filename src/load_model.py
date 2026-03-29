from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "PyTorch is required for model loading. Install `torch` before running the experiment."
    ) from exc

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "transformers is required for model loading. Install `transformers` and `sentencepiece`."
    ) from exc


def select_device(device: str | None = None) -> torch.device:
    """Pick an execution device, defaulting to the fastest available backend."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str = "camembert-base", device: str | None = None):
    """Load a tokenizer/model pair configured to expose hidden states and attentions."""
    torch_device = select_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    config = AutoConfig.from_pretrained(model_name)
    model_kwargs = {}

    # Newer transformers versions may default to SDPA attention, which does not
    # expose attention weights for some encoder architectures. Force eager mode
    # when the config supports it so `output_attentions=True` behaves as expected.
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    else:
        model_kwargs["attn_implementation"] = "eager"

    model = AutoModel.from_pretrained(model_name, config=config, **model_kwargs)
    model.to(torch_device)
    model.eval()
    return tokenizer, model, torch_device
