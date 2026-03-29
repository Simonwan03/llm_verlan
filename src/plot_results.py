from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "matplotlib is required for plotting. Install `matplotlib` before running plot_results.py."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Plot verlan probing experiment outputs.")
    parser.add_argument(
        "--results-csv",
        default="outputs/verlan_probe/experiment_results.csv",
        help="Layer-wise results CSV produced by run_experiment.py.",
    )
    parser.add_argument(
        "--tokenization-csv",
        default="outputs/verlan_probe/tokenization_results.csv",
        help="Tokenization CSV produced by run_experiment.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/verlan_probe/figures",
        help="Directory where plots and summary tables will be saved.",
    )
    return parser.parse_args()


def save_subword_summary(tokenization_df: pd.DataFrame, output_dir: Path):
    grouped = (
        tokenization_df.groupby(["verlan", "standard", "variant"])["num_target_subwords"]
        .mean()
        .reset_index()
    )
    summary = (
        grouped.pivot(index=["verlan", "standard"], columns="variant", values="num_target_subwords")
        .rename_axis(None, axis=1)
        .rename(
            columns={
                "verlan": "verlan_subwords",
                "standard": "standard_subwords",
            }
        )
        .reset_index()
    )
    if "verlan_subwords" not in summary.columns:
        summary["verlan_subwords"] = 0.0
    if "standard_subwords" not in summary.columns:
        summary["standard_subwords"] = 0.0
    summary["subword_delta_verlan_minus_standard"] = (
        summary["verlan_subwords"] - summary["standard_subwords"]
    )
    summary.to_csv(output_dir / "subword_fragmentation_summary.csv", index=False)


def plot_word_similarity(results_df: pd.DataFrame, output_dir: Path):
    plot_df = results_df.groupby("layer", as_index=False)["word_similarity"].mean()
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["layer"], plot_df["word_similarity"], marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Average word-level cosine similarity")
    plt.title("Layer-wise word similarity: verlan vs standard")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "word_similarity_by_layer.png", dpi=200)
    plt.close()


def plot_sentence_similarity(results_df: pd.DataFrame, output_dir: Path):
    plot_df = results_df.groupby("layer", as_index=False)["sentence_similarity"].mean()
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["layer"], plot_df["sentence_similarity"], marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Average sentence-level cosine similarity")
    plt.title("Layer-wise sentence drift: verlan vs standard")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "sentence_similarity_by_layer.png", dpi=200)
    plt.close()


def plot_attention_breakdown(results_df: pd.DataFrame, output_dir: Path):
    attn_df = (
        results_df[results_df["layer"] > 0]
        .groupby("layer", as_index=False)[
            [
                "intra_verlan",
                "intra_standard",
                "context_verlan",
                "context_standard",
                "incoming_verlan",
                "incoming_standard",
            ]
        ]
        .mean()
    )

    plt.figure(figsize=(8, 5))
    plt.plot(attn_df["layer"], attn_df["intra_verlan"], marker="o", label="Intra-word (verlan)")
    plt.plot(attn_df["layer"], attn_df["intra_standard"], marker="o", label="Intra-word (standard)")
    plt.xlabel("Layer")
    plt.ylabel("Attention ratio")
    plt.title("Layer-wise intra-word attention")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "intra_word_attention_by_layer.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(attn_df["layer"], attn_df["context_verlan"], marker="o", label="Context (verlan)")
    plt.plot(attn_df["layer"], attn_df["context_standard"], marker="o", label="Context (standard)")
    plt.xlabel("Layer")
    plt.ylabel("Attention ratio")
    plt.title("Layer-wise context attention")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "context_attention_by_layer.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(attn_df["layer"], attn_df["incoming_verlan"], marker="o", label="Incoming (verlan)")
    plt.plot(attn_df["layer"], attn_df["incoming_standard"], marker="o", label="Incoming (standard)")
    plt.xlabel("Layer")
    plt.ylabel("Attention ratio")
    plt.title("Layer-wise incoming attention from context")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "incoming_attention_by_layer.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(args.results_csv)
    tokenization_df = pd.read_csv(args.tokenization_csv)

    save_subword_summary(tokenization_df, output_dir)
    plot_word_similarity(results_df, output_dir)
    plot_sentence_similarity(results_df, output_dir)
    plot_attention_breakdown(results_df, output_dir)

    print(f"Saved figures and summary tables to {output_dir}")


if __name__ == "__main__":
    main()
