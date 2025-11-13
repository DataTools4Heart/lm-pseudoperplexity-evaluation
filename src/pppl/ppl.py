# %%
import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

logger = None  # Global logger variable


def setup_logger(path: str = "logs"):
    global logger
    if logger is None:
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure the directory exists
        logger = logging.getLogger("custom_logger")
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(os.path.join(path, "log.txt"))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(stream_handler)


def sliding_window_reshape_batch(
    batch: Dict[str, List[List[int]]], window_size: int, stride: int, pad_token_id: int
):
    """
    Applies sliding window reshaping to a batch of sequences.
    Input:

    - batch: Batch of sequences with shape (batch_size, sequence_length). Useful for hf datasets parallelization.
    - window_size: Size of the sliding window.
    - stride: Stride of the sliding window.
    - pad_token_id: Token ID for padding.

    Output:

    - Dictionary with the reshaped input_ids and attention_mask. This is a standard format for hf datasets.

    """
    batch_input_ids = batch["input_ids"]

    all_input_ids = []
    all_attention_masks = []

    for input_ids in batch_input_ids:
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(
            0
        )  # Ensure 2D shape

        n = input_ids.size(-1)  # Sequence length
        n_windows = (n - 1) // stride + 1  # Compute number of windows

        # Initialize output tensors
        input_ids_out = torch.full(
            (n_windows, window_size), pad_token_id, dtype=torch.long
        )
        attention_mask_out = torch.zeros((n_windows, window_size), dtype=torch.long)

        # Fill the windows
        for i, start in enumerate(range(0, n, stride)):
            end = min(start + window_size, n)
            length = end - start
            input_ids_out[i, :length] = input_ids[0, start:end]
            attention_mask_out[i, :length] = 1  # Mask only valid tokens

        all_input_ids.append(input_ids_out)
        all_attention_masks.append(attention_mask_out)

    return {"input_ids": all_input_ids, "attention_mask": all_attention_masks}


def create_iterative_masking(input_id: List[int], mask_token: int, pad_token_id: int):
    """
    Receives a list of integers, duplicated it and replaces each element with a mask_token.

    - input_id: List of integers.
    - mask_token: Token to replace the elements. If -999, no replacement is done. Useful for labels.
    - pad_token_id: Token ID for padding.

    Output:

    - Tuple with the masked sequence and the attention mask.
    """

    if mask_token is None:
        raise ValueError("mask_token cannot be None.")

    input_id = torch.tensor(input_id)  # Convert to tensor
    attention_mask = torch.ones_like(input_id)  # Create attention mask
    attention_mask[input_id == pad_token_id] = 0  # Set padding tokens to 0

    n = input_id.shape[0]  # Number
    n_pad = input_id[input_id == pad_token_id].shape[0]  # Number of padding tokens

    masked_sequence = input_id.repeat(n - n_pad, 1)
    attention_mask = attention_mask.repeat(n - n_pad, 1)

    if mask_token != -999:
        # Replace diagonal elements with mask_token
        masked_sequence.fill_diagonal_(mask_token)

    return masked_sequence, attention_mask


def multiple_masked_ids(
    batch: Dict[str, List[List[int]]], mask_token_id: int, pad_token_id: int
):
    """
    Applies multiple masking to a batch of sequences.
    Input:

    - batch: Batch of sequences with shape (batch_size, n_windows, window_size). Useful for hf datasets parallelization.
    - mask_token_id: Token ID for masking.
    - pad_token_id: Token ID for padding. Set to 0 by default.

    Output:

    - Dictionary with the masked input_ids and attention_mask. This is a standard format for hf datasets.

    """

    input_ids = batch["input_ids"]

    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    for input_id in input_ids:
        ls_rows_input_id = []
        ls_rows_attention_mask = []
        ls_labels = []
        for row in input_id:
            input_id_out, attention_mask_out = create_iterative_masking(
                row, mask_token=mask_token_id, pad_token_id=pad_token_id
            )
            labels, _ = create_iterative_masking(
                row, mask_token=-999, pad_token_id=pad_token_id
            )
            ls_rows_input_id.extend(input_id_out)
            ls_rows_attention_mask.extend(attention_mask_out)
            ls_labels.extend(labels)

        all_input_ids.append(ls_rows_input_id)
        all_attention_masks.append(ls_rows_attention_mask)
        all_labels.append(ls_labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_input_ids,
    }


def build_tensors_from_df(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    windows_size: int,
    stride: int,
    mask_token_id: int,
    pad_token_id: int,
):
    """
    Builds tensors from a DataFrame with text data.

    - df: DataFrame with text data. It must contain a column named "text".
    - tokenizer: Hugging Face tokenizer.

    Output:

    - Tuple with input_ids, attention_mask and labels tensors.
    """

    # Convert DataFrame to Hugging Face Dataset and tokenize the text
    dataset = Dataset.from_pandas(df).map(lambda x: tokenizer(x["text"]), batched=True)

    # Apply sliding window reshape in batched mode
    dataset = dataset.map(
        lambda batch: sliding_window_reshape_batch(
            batch, windows_size, stride, pad_token_id
        ),
        batched=True,
        remove_columns=["input_ids", "attention_mask"],  # Remove old columns
    )

    # Apply multiple masking to the dataset
    dataset = dataset.map(
        lambda batch: multiple_masked_ids(batch, mask_token_id, pad_token_id),
        batched=True,
        remove_columns=["input_ids", "attention_mask"],  # Remove old columns
    )

    # concatenate efficiently all the input_ids and attention_mask
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tensor_ids = torch.cat(dataset["input_ids"], dim=0)
    tensor_attention_mask = torch.cat(dataset["attention_mask"], dim=0)
    tensor_labels = torch.cat(dataset["labels"], dim=0)

    return tensor_ids, tensor_attention_mask, tensor_labels


if __name__ == "__main__":
    # python ppl.py \
    #   --model "/gpfs/projects/bsc14/abecerr1/hub/models--PlanTL-GOB-ES--roberta-base-biomedical-clinical-es/snapshots/c6bfaa3cc4453dc6d947d279e3905c7083663af1/" \
    #   --csv_path "data/data/paraclite.csv" \
    #   --language "es"

    parser = argparse.ArgumentParser(
        description="Compute pseudo-perplexity of a model."
    )

    # Mandatory
    parser.add_argument(
        "--model_name", type=str, help="Hugging Face model name or path."
    )
    parser.add_argument(
        "--csv_path", type=str, help="Path to the CSV file containing text data."
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["sv", "es", "en", "cz", "it", "nl", "ro"],
        help="Language column to use from CSV.",
    )
    parser.add_argument(
        "--mask_token_id",
        type=int,
        help="Check your tokenizer mask token ID before running.",
    )
    parser.add_argument(
        "--pad_token_id",
        type=int,
        help="Check your tokenizer pad token ID before running.",
    )

    # Optional
    parser.add_argument(
        "--windows_size", type=int, default=256, help="Sliding window size."
    )
    parser.add_argument(
        "--stride", type=int, default=256, help="Sliding window stride."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Model inference batch size."
    )
    parser.add_argument(
        "--output_path", type=str, default="output", help="Path to the output log file."
    )

    args = parser.parse_args()

    # Variables definition
    model_name = args.model_name
    csv_path = args.csv_path
    language = args.language
    device = "cuda"  # if torch.cuda.is_available() else "cpu"
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    windows_size = (
        args.windows_size
        if args.windows_size is not None
        else tokenizer.model_max_length
    )
    stride = args.stride if args.stride is not None else tokenizer.model_max_length // 2

    if args.pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = args.pad_token_id

    if args.mask_token_id is None:
        mask_token_id = tokenizer.mask_token_id
    else:
        mask_token_id = args.mask_token_id

    batch_size = args.batch_size
    output_path = os.path.join(
        args.output_path, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(output_path, exist_ok=True)

    setup_logger(path=output_path)

    logger.info(f"Model: {model_name}")
    logger.info(f"Masking ID: {mask_token_id}")
    logger.info(f"Pad ID: {pad_token_id}")
    logger.info(f"CSV Path: {csv_path}")
    logger.info(f"Language: {language}")
    logger.info(f"Device: {device}")
    logger.info(f"Max Length: {tokenizer.model_max_length}")
    logger.info(f"Windows Size: {windows_size}")
    logger.info(f"Stride: {stride}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Output Path: {output_path}")

    # Load CSV and extract text
    logger.info("Loading CSV and tokenizing text...")
    df = pd.read_csv(csv_path)
    if language not in df.columns:
        raise ValueError(
            f"The specified language column '{language}' is not found in the CSV."
        )

    df_lang = (
        df.groupby("doc_name")
        .apply(lambda x: x[language].str.cat(sep="\n"), include_groups=False)
        .reset_index()
    )
    df_lang.columns = ["doc_name", "text"]

    logger.info(f"Number of documents: {df_lang.shape[0]}")

    logger.info("Building tensors...")
    tensor_ids, tensor_attention_mask, tensor_labels = build_tensors_from_df(
        df_lang, tokenizer, windows_size, stride, mask_token_id, pad_token_id
    )

    logger.info("Computing pseudo-perplexity...")
    # run the model on tensor_ids and tensor_attention_mask but in batches
    n = tensor_ids.shape[0]
    # n = 256 # For testing purposes
    ls_nll = []
    d_analysis = {}
    for i in tqdm(range(0, n, batch_size)):
        batch_input_ids = tensor_ids[i : i + batch_size].to(device)
        batch_attention_mask = tensor_attention_mask[i : i + batch_size].to(device)
        batch_tensor_labels = tensor_labels[i : i + batch_size].to(device)

        with torch.no_grad():
            output = model(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_tensor_labels,
            )
            neg_log_likelihood = output.loss  # Model loss corresponds to NLL

        ls_nll.append(neg_log_likelihood.item())
        d_analysis[i] = {
            "nll": neg_log_likelihood.item(),
            "input_ids": batch_input_ids.cpu().tolist(),
        }

    ls_nll = torch.tensor(ls_nll)
    nll_mean = ls_nll.mean()
    nll_median = ls_nll.median()
    ppl = torch.exp(nll_mean)
    ppl_median = torch.exp(nll_median)

    logger.info(f"Mean NLL: {nll_mean:.4f}")
    logger.info(f"PPL: {ppl:.4f}")

    logger.info("Saving output...")
    ls_nll = ls_nll.cpu().numpy()
    output = pd.DataFrame({"NLL": ls_nll})
    output.to_csv(os.path.join(output_path, "nll.csv"), index=False)

    with open(os.path.join(output_path, "analysis.json"), "w") as f:
        json.dump(d_analysis, f)

    import json

    json.dump(
        {
            "mean_nll": nll_mean.item(),
            "median_nll": nll_median.item(),
            "ppl": ppl.item(),
            "ppl_median": ppl_median.item(),
        },
        open(os.path.join(output_path, "metrics.json"), "w"),
    )

    logger.info("Done!")
