"""Span-in-context pseudo-perplexity backends for medical NER spans.

Extends PPLBackend and MiniconsStridedBackend with get_span_perplexity():
every token within a character span is masked and scored one at a time using
the full surrounding document as context. Tokens outside the span are never
masked but remain visible to the model as attending context.

Overlapping windows: consistent with the parent backends, a span token that
falls inside multiple overlapping windows is scored once per window. All
(token, window) pairs contribute equally to the final mean NLL.

Typical usage
-------------
    from pppl.backends_medical_span import SpanPPLBackend, SpanMiniconsStridedBackend
    from pppl.span_utils import load_predictions, load_paraclite_docs, build_span_dataset

    docs   = load_paraclite_docs("data/paraclite.csv", language="nl")
    spans  = load_predictions("paraclite_inference_results/nl_multilabel_xlm_roberta_large_all_entities_predictions.tsv")
    df     = build_span_dataset(spans, docs)

    backend = SpanPPLBackend("path/to/robbert", device="cuda", windows_size=508, stride=256, batch_size=8)
    df["ppl"] = backend.score_spans(df, show_progress=True)
"""

import math
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from pppl.backends import PPLBackend, MiniconsStridedBackend
from pppl.span_utils import chars_to_token_indices


class SpanPPLBackend(PPLBackend):
    """PPLBackend extended with span-in-context pseudo-perplexity scoring.

    All constructor arguments are identical to PPLBackend. The parent's
    get_perplexity() method is unchanged and still scores full documents.
    """

    def get_span_perplexity(
        self,
        text: str,
        start_char: int,
        end_char: int,
    ) -> float:
        """Pseudo-perplexity restricted to a character span, scored in context.

        The full document is tokenized (with special tokens, matching the
        parent's behaviour) and split into sliding windows of self.windows_size
        with stride self.stride. For each window that contains at least one
        span token, every span token in that window is masked individually
        while the rest of the window provides context. The final score is
        exp(mean NLL over all span-token/window pairs).

        Args:
            text: Full document text (e.g. from build_span_dataset's 'doc_text').
            start_char: Inclusive start of the character span.
            end_char: Exclusive end of the character span (i.e. text[start:end]).

        Returns:
            Span pseudo-perplexity, or float('nan') when no maskable span
            tokens are found (e.g. the span contains only punctuation or
            falls outside the document).

        Note:
            Requires a fast tokenizer (offset_mapping support). All RoBERTa-
            based models (robbert, xlm-roberta, etc.) satisfy this automatically.
        """
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            truncation=False,
        )
        token_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        # Positions in the with-specials token sequence that overlap the span.
        # chars_to_token_indices already excludes zero-length tokens, which
        # are the special tokens ([CLS]/[SEP]) with offset (0, 0). The
        # special_ids check below is a belt-and-suspenders guard for tokenizers
        # that assign non-zero offsets to their special tokens.
        span_positions = chars_to_token_indices(offsets, start_char, end_char)
        if self.special_ids:
            span_positions -= {
                i for i, tid in enumerate(token_ids) if tid in self.special_ids
            }

        if not span_positions:
            return float("nan")

        n = len(token_ids)
        token_ids_t = torch.tensor(token_ids, dtype=torch.long)

        total_nll = 0.0
        total_scored = 0

        for window_start in range(0, n, self.stride):
            window_end = min(window_start + self.windows_size, n)

            window_span = {p for p in span_positions if window_start <= p < window_end}
            if not window_span:
                continue

            window_len = window_end - window_start
            window_ids = token_ids_t[window_start:window_end]

            # Pad the last (shorter) window to windows_size.
            if window_len < self.windows_size:
                pad = torch.full(
                    (self.windows_size - window_len,),
                    self.pad_token_id,
                    dtype=torch.long,
                )
                window_ids = torch.cat([window_ids, pad])
                attn_mask = torch.cat([
                    torch.ones(window_len, dtype=torch.long),
                    torch.zeros(self.windows_size - window_len, dtype=torch.long),
                ])
            else:
                attn_mask = torch.ones(self.windows_size, dtype=torch.long)

            # Build one row per span token: mask that position, label it with
            # the original token id so HuggingFace's CrossEntropyLoss scores it.
            rel_positions = sorted(p - window_start for p in window_span)
            n_span = len(rel_positions)

            batch_ids = window_ids.unsqueeze(0).expand(n_span, -1).clone()
            batch_mask = attn_mask.unsqueeze(0).expand(n_span, -1).clone()
            batch_labels = torch.full(
                (n_span, self.windows_size), -100, dtype=torch.long
            )
            for row, rel_pos in enumerate(rel_positions):
                original_id = batch_ids[row, rel_pos].item()
                batch_ids[row, rel_pos] = self.mask_token_id
                batch_labels[row, rel_pos] = original_id

            # Forward in sub-batches of self.batch_size to bound GPU memory.
            for i in range(0, n_span, self.batch_size):
                bi = batch_ids[i : i + self.batch_size].to(self.device)
                bm = batch_mask[i : i + self.batch_size].to(self.device)
                bl = batch_labels[i : i + self.batch_size].to(self.device)

                with torch.no_grad():
                    out = self.model(bi, attention_mask=bm, labels=bl)

                # output.loss is mean NLL over non-ignored positions in the
                # batch. Each row has exactly one labeled position, so this
                # equals the mean NLL over the sub-batch's span tokens.
                sub_bs = bi.size(0)
                total_nll += out.loss.item() * sub_bs
                total_scored += sub_bs

        if total_scored == 0:
            return float("nan")
        return math.exp(total_nll / total_scored)

    def score_spans(
        self,
        spans_df: pd.DataFrame,
        show_progress: bool = False,
    ) -> pd.Series:
        """Score every row of a span DataFrame in context.

        Args:
            spans_df: DataFrame with columns 'doc_text' (full document string),
                'start_span', and 'end_span'. Typically the output of
                build_span_dataset().
            show_progress: Display a tqdm progress bar over spans.

        Returns:
            pd.Series of perplexity scores with the same index as spans_df.
        """
        rows = spans_df.iterrows()
        if show_progress:
            rows = tqdm(rows, total=len(spans_df), desc="SpanPPLBackend")

        scores = [
            self.get_span_perplexity(row["doc_text"], row["start_span"], row["end_span"])
            for _, row in rows
        ]
        return pd.Series(scores, index=spans_df.index)


class SpanMiniconsStridedBackend(MiniconsStridedBackend):
    """MiniconsStridedBackend extended with span-in-context pseudo-perplexity.

    All constructor arguments are identical to MiniconsStridedBackend. The
    parent's get_perplexity() method is unchanged and still scores full documents.
    """

    def _score_span_tokens(
        self,
        window_enc,
        raw_offsets: List[Tuple[int, int]],
        span_lp_indices: List[int],
    ) -> torch.Tensor:
        """Score a subset of tokens in a window, masking each one individually.

        Runs exactly len(span_lp_indices) forward passes — one per span token —
        instead of one per content token in the window. For a 4-token span in a
        506-token window this is ~125× fewer forward passes than prepare_text.

        Args:
            window_enc: BatchEncoding for the window (must include attention_mask
                and offset_mapping).
            raw_offsets: window_enc['offset_mapping'][0] — used to locate each
                content token's position within the full sequence.
            span_lp_indices: 0-indexed positions into the content-token
                subsequence (special tokens excluded).

        Returns:
            1-D CPU tensor of log-probabilities, one entry per span token.
        """
        input_ids = torch.tensor(window_enc["input_ids"][0], dtype=torch.long)
        attn_mask = torch.tensor(window_enc["attention_mask"][0], dtype=torch.long)

        # Map content-token index j → full-sequence position (skipping specials).
        content_seq_positions = [
            pos for pos, (ws, we) in enumerate(raw_offsets) if we > ws
        ]

        mask_id = self._scorer.tokenizer.mask_token_id
        n_span = len(span_lp_indices)

        # word_ids() maps each sequence position to a word index (None for
        # special tokens). Used to implement within_word_l2r: when scoring
        # token at seq_pos, all subsequent tokens belonging to the same word
        # are also masked so the model cannot peek at them.
        word_ids = window_enc.word_ids(batch_index=0)

        # One row per span token: mask that position, keep everything else intact
        # so the model sees the full surrounding context.
        batch_ids  = input_ids.unsqueeze(0).expand(n_span, -1).clone()
        batch_attn = attn_mask.unsqueeze(0).expand(n_span, -1).clone()
        target_ids = torch.zeros(n_span, dtype=torch.long)
        target_pos = torch.zeros(n_span, dtype=torch.long)

        for row, j in enumerate(span_lp_indices):
            seq_pos = content_seq_positions[j]
            target_ids[row] = input_ids[seq_pos]
            target_pos[row] = seq_pos
            batch_ids[row, seq_pos] = mask_id
            # Mask all subsequent tokens that belong to the same word
            # (within_word_l2r: left-to-right factorization within each word).
            target_word_id = word_ids[seq_pos]
            if target_word_id is not None:
                for other_pos in range(seq_pos + 1, len(word_ids)):
                    if word_ids[other_pos] == target_word_id:
                        batch_ids[row, other_pos] = mask_id

        device = self._scorer.model.device
        out_chunks: List[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, n_span, self.model_batch_size):
                bi = batch_ids[i : i + self.model_batch_size].to(device)
                bm = batch_attn[i : i + self.model_batch_size].to(device)
                ti = target_ids[i : i + self.model_batch_size].to(device)
                tp = target_pos[i : i + self.model_batch_size].to(device)

                logits  = self._scorer.model(bi, attention_mask=bm).logits  # (B, T, V)
                rows    = torch.arange(logits.shape[0], device=device)
                tgt_lgt = logits[rows, tp]                                  # (B, V)
                log_p   = tgt_lgt - tgt_lgt.logsumexp(-1, keepdim=True)    # (B, V)
                out_chunks.append(log_p[rows, ti].detach().cpu())           # (B,)

        return torch.cat(out_chunks)

    def get_span_perplexity(
        self,
        text: str,
        start_char: int,
        end_char: int,
    ) -> float:
        """Pseudo-perplexity restricted to a character span, scored in context.

        Tokenizes the full document without special tokens to obtain character
        offsets and window boundaries, then slides the same content window as
        get_perplexity() across the token sequence. For each window containing
        at least one span token, _score_span_tokens() masks and scores only
        those span tokens (one forward pass per span token). The final score is
        exp(mean NLL over all span-token/window pairs).

        Args:
            text: Full document text (e.g. from build_span_dataset's 'doc_text').
            start_char: Inclusive start of the character span.
            end_char: Exclusive end of the character span.

        Returns:
            Span pseudo-perplexity, or float('nan') when no span tokens are found.
        """
        full = self.tokenizer(
            text,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        full_offsets = full["offset_mapping"]
        n = len(full_offsets)

        if n == 0:
            return float("nan")

        if not chars_to_token_indices(full_offsets, start_char, end_char):
            return float("nan")

        all_span_logprobs: List[torch.Tensor] = []
        start = 0

        while start < n:
            end = min(start + self._content_window, n)
            char_start = full_offsets[start][0]
            char_end   = full_offsets[end - 1][1]
            chunk_text = text[char_start:char_end]

            # Single tokenizer call: offset_mapping for span mapping +
            # attention_mask for scoring — no second call needed.
            window_enc = self.tokenizer(
                [chunk_text],
                padding=False,
                truncation=True,
                max_length=self.windows_size,
                return_attention_mask=True,
                return_offsets_mapping=True,
            )
            raw_offsets = window_enc["offset_mapping"][0]

            content_doc_offsets = [
                (char_start + ws, char_start + we)
                for ws, we in raw_offsets
                if we > ws
            ]
            span_lp_indices = [
                j
                for j, (doc_s, doc_e) in enumerate(content_doc_offsets)
                if doc_s < end_char and doc_e > start_char
            ]

            if span_lp_indices:
                all_span_logprobs.append(
                    self._score_span_tokens(window_enc, raw_offsets, span_lp_indices)
                )

            if start + self._content_window >= n:
                break
            start += self.stride

        if not all_span_logprobs:
            return float("nan")

        all_lp = torch.cat(all_span_logprobs)
        return math.exp(-all_lp.mean().item())

    def score_spans(
        self,
        spans_df: pd.DataFrame,
        show_progress: bool = False,
    ) -> pd.Series:
        """Score every row of a span DataFrame in context.

        Args:
            spans_df: DataFrame with columns 'doc_text' (full document string),
                'start_span', and 'end_span'. Typically the output of
                build_span_dataset().
            show_progress: Display a tqdm progress bar over spans.

        Returns:
            pd.Series of perplexity scores with the same index as spans_df.
        """
        rows = spans_df.iterrows()
        if show_progress:
            rows = tqdm(rows, total=len(spans_df), desc="SpanMiniconsStridedBackend")

        scores = [
            self.get_span_perplexity(row["doc_text"], row["start_span"], row["end_span"])
            for _, row in rows
        ]
        return pd.Series(scores, index=spans_df.index)
