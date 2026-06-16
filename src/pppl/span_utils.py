"""Utilities for building span-level datasets and mapping character spans to token indices."""

import csv
import warnings
from typing import List, Set, Tuple

import pandas as pd

VALID_LABELS = {"DISEASE", "PROCEDURE", "SYMPTOM", "MEDICATION"}


def chars_to_token_indices(
    offsets: List[Tuple[int, int]],
    start_char: int,
    end_char: int,
) -> Set[int]:
    """Return token indices (0-based) whose character range overlaps [start_char, end_char).

    Zero-length tokens — e.g. special tokens with offset (0, 0) — are always excluded.

    Args:
        offsets: List of (char_start, char_end) pairs from a tokenizer's offset_mapping.
        start_char: Inclusive start of the character span.
        end_char: Exclusive end of the character span.

    Returns:
        Set of 0-indexed token positions that overlap the span.
    """
    return {
        i
        for i, (s, e) in enumerate(offsets)
        if e > s and s < end_char and e > start_char
    }


def load_predictions(tsv_path: str) -> pd.DataFrame:
    """Load a NER prediction TSV, keeping only rows with recognised entity labels.

    Handles quoted multiline text spans (CSV-style quoting inside TSV files).
    Rows where the label is not in VALID_LABELS are silently dropped — these
    are usually parsing artifacts produced by embedded newlines in span text.

    Args:
        tsv_path: Path to a ``*_predictions.tsv`` file from paraclite_inference_results.

    Returns:
        DataFrame with columns: filename, label, start_span, end_span, text.
    """
    df = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_MINIMAL)
    df = df[df["label"].isin(VALID_LABELS)].copy()
    df["start_span"] = df["start_span"].astype(int)
    df["end_span"] = df["end_span"].astype(int)
    return df.reset_index(drop=True)


def load_paraclite_docs(csv_path: str, language: str) -> pd.DataFrame:
    """Aggregate paraclite.csv segments into per-document texts for one language.

    Concatenates segments with '\\n' in seg_id order, matching the aggregation
    used during NER inference so that start_span/end_span offsets in prediction
    files index correctly into the returned text.

    Args:
        csv_path: Path to paraclite.csv.
        language: Column name to use as document text (e.g. 'nl', 'en', 'cs').

    Returns:
        DataFrame with columns ['doc_name', 'text'], one row per document.
        doc_name values have the '.txt' suffix stripped to match prediction
        filenames (e.g. 'ro_patient_4' instead of 'ro_patient_4.txt').
    """
    df = pd.read_csv(csv_path).sort_values(["doc_name", "seg_id"])
    agg = (
        df.groupby("doc_name")
        .apply(lambda x: x[language].str.cat(sep="\n"), include_groups=False)
        .reset_index()
    )
    agg.columns = ["doc_name", "text"]
    agg["doc_name"] = agg["doc_name"].str.replace(".txt", "", regex=False)
    return agg


def build_span_dataset(
    predictions_df: pd.DataFrame,
    docs_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join NER predictions with their full document texts.

    Args:
        predictions_df: Output of load_predictions(). Must have 'filename'.
        docs_df: Output of load_paraclite_docs(). Must have 'doc_name', 'text'.

    Returns:
        Merged DataFrame with all prediction columns plus 'doc_text' holding the
        full document string. Rows with no matching document are dropped with a
        warning.
    """
    merged = predictions_df.merge(
        docs_df.rename(columns={"text": "doc_text"}),
        left_on="filename",
        right_on="doc_name",
        how="left",
    )
    n_missing = merged["doc_text"].isna().sum()
    if n_missing:
        warnings.warn(
            f"build_span_dataset: {n_missing} span(s) could not be matched to a "
            "document and will be dropped. Check that docs_df covers all filenames "
            "in predictions_df.",
            stacklevel=2,
        )
    return merged.dropna(subset=["doc_text"]).reset_index(drop=True)
