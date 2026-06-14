"""
Backend wrappers for perplexity scoring using lmppl and minicons libraries.

This module provides a unified interface for computing perplexity scores
using different backends:
- lmppl: https://github.com/asahi417/lmppl
- minicons: https://github.com/kanishkamisra/minicons
- pppl:    custom iterative-masking pseudo-perplexity implementation (see ppl.py)
"""

import math
import warnings
from typing import List, Literal, Optional, Tuple, Union

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from pppl.ppl import build_tensors_from_df

# Hugging Face uses a sentinel like 1e30 for `model_max_length` when the
# tokenizer does not actually know the model's max length. Anything above
# this threshold is treated as "unknown" rather than a real limit.
_HF_MAX_LENGTH_SENTINEL = 1_000_000


class PPLBackend:
    """Backend using a custom iterative-masking pseudo-perplexity implementation.

    Implements the same scoring procedure as ``ppl.py``: each text is
    tokenized, reshaped into sliding windows, every non-padding token is
    masked one at a time, and the model's mean negative log-likelihood over
    the masked positions is exponentiated to obtain the pseudo-perplexity.

    Supports masked language models (e.g., BERT, RoBERTa, DeBERTa).

    Example:
        >>> backend = PPLBackend('bert-base-uncased', device='cpu')
        >>> texts = ['I am happy.', 'I am sad.']
        >>> scores = backend.get_perplexity(texts)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        windows_size: int = 256,
        stride: int = 256,
        batch_size: int = 32,
        mask_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        exclude_special_tokens: bool = True,
        show_progress: bool = False,
    ):
        """Initialize the custom pseudo-perplexity backend.

        Args:
            model_name: Hugging Face model name or path for a masked LM.
            device: Device to run the model on ('cpu' or 'cuda').
            windows_size: Sliding window size used to chunk long inputs.
            stride: Sliding window stride.
            batch_size: Batch size used during model inference.
            mask_token_id: Optional override for the tokenizer's mask token id.
            pad_token_id: Optional override for the tokenizer's pad token id.
            exclude_special_tokens: When True (default), structural special
                tokens (``[CLS]``/``[SEP]``/``[BOS]``/``[EOS]``/pad) are not
                masked or scored, matching minicons' convention. They remain
                in the sequence and are still attended to. Set to False to
                score every non-pad position (the original ``ppl.py``
                behavior).
            show_progress: Whether to display a tqdm progress bar during inference.
        """
        self.model_name = model_name
        self.device = device
        self.windows_size = windows_size
        self.stride = stride
        self.batch_size = batch_size
        self.show_progress = show_progress

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()

        self.mask_token_id = (
            mask_token_id if mask_token_id is not None else self.tokenizer.mask_token_id
        )
        self.pad_token_id = (
            pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        )

        if self.mask_token_id is None:
            raise ValueError(
                "Tokenizer has no mask_token_id; pass mask_token_id explicitly."
            )
        if self.pad_token_id is None:
            raise ValueError(
                "Tokenizer has no pad_token_id; pass pad_token_id explicitly."
            )

        # Structural framing tokens to exclude from masking/scoring. We use
        # only the structural specials (cls/sep/bos/eos/pad) and deliberately
        # NOT unk or mask: unk represents real (if unknown) content, and the
        # mask token never appears in the original input. This matches what
        # minicons excludes (pad/cls/sep).
        self.special_ids: Optional[set] = None
        if exclude_special_tokens:
            candidate_ids = {
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                self.pad_token_id,
            }
            self.special_ids = {tid for tid in candidate_ids if tid is not None}

    def _score_single(self, text: str) -> float:
        """Compute pseudo-perplexity for a single text."""
        df = pd.DataFrame({"text": [text]})
        tensor_ids, tensor_attention_mask, tensor_labels = build_tensors_from_df(
            df,
            self.tokenizer,
            self.windows_size,
            self.stride,
            self.mask_token_id,
            self.pad_token_id,
            special_ids=self.special_ids,
        )

        n = tensor_ids.shape[0]
        if n == 0:
            return float("nan")

        total_nll = 0.0
        total_samples = 0

        iterator = range(0, n, self.batch_size)
        if self.show_progress:
            iterator = tqdm(iterator)

        for i in iterator:
            batch_input_ids = tensor_ids[i : i + self.batch_size].to(self.device)
            batch_attention_mask = tensor_attention_mask[i : i + self.batch_size].to(
                self.device
            )
            batch_labels = tensor_labels[i : i + self.batch_size].to(self.device)

            with torch.no_grad():
                output = self.model(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels,
                )

            batch_samples = batch_input_ids.size(0)
            total_nll += output.loss.item() * batch_samples
            total_samples += batch_samples

        mean_nll = total_nll / total_samples
        return math.exp(mean_nll)

    def get_perplexity(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        """Compute pseudo-perplexity for one or more texts.

        Args:
            text: A single string or list of strings to score.

        Returns:
            A single perplexity score or list of scores.
        """
        if isinstance(text, str):
            return self._score_single(text)
        return [self._score_single(t) for t in text]


class LmpplBackend:
    """Backend using lmppl library for perplexity scoring.

    Supports masked language models (e.g., DeBERTa, BERT, RoBERTa).

    Example:
        >>> backend = LmpplBackend('microsoft/deberta-v3-small')
        >>> texts = ['I am happy.', 'I am sad.']
        >>> scores = backend.get_perplexity(texts)
    """

    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 32):
        """Initialize the lmppl backend.

        Args:
            model_name: Hugging Face model name or path for a masked LM.
            max_length: Maximum tokenized sequence length passed to
                ``lmppl.MaskedLM``. Inputs longer than this are silently
                truncated by lmppl to their first ``max_length`` tokens
                (it does not perform sliding-window scoring).
            batch_size: Inference batch size forwarded to
                ``lmppl.MaskedLM.get_perplexity``.
        """
        try:
            import lmppl
        except ImportError:
            raise ImportError(
                "lmppl is required for this backend. Install it with: pip install lmppl"
            )

        # NOTE: pass max_length as a keyword. lmppl.MaskedLM's positional
        # signature is (model, use_auth_token, max_length, ...), so a
        # positional `max_length` would silently land in `use_auth_token`
        # and leave the scorer's internal `max_length` as None -- which
        # causes IndexError on inputs longer than the model's max length.
        self.scorer = lmppl.MaskedLM(model_name, max_length=max_length)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

    def get_perplexity(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        """Compute perplexity for one or more texts.

        Args:
            text: A single string or list of strings to score.

        Returns:
            A single perplexity score or list of scores.
        """
        if isinstance(text, str):
            text = [text]
            return self.scorer.get_perplexity(text)[0]
        return self.scorer.get_perplexity(text, batch_size=self.batch_size)


class LmpplStridedBackend:
    """Lmppl backend with text-level striding for full-document scoring.

    :class:`LmpplBackend` silently truncates inputs to the first
    ``max_length`` tokens (lmppl does no windowing of its own). This class
    wraps lmppl with a *text-level* sliding window: each document is
    pre-split into overlapping chunks of approximately ``windows_size``
    tokens using the model's fast tokenizer (via ``offset_mapping``), each
    chunk is scored independently by lmppl, and per-chunk
    pseudo-perplexities are aggregated into a single document score.

    Why pre-split the *text* (not the token ids):
        - Lmppl's public API takes raw strings; it does not accept
          token-id batches. To stay inside that API we either have to
          detokenize sliced token ids (which is not a faithful round-trip
          for subword tokenizers) or pick substrings of the original text
          that happen to align with tokenizer boundaries. ``offset_mapping``
          on the fast tokenizer gives us exactly that.

    Aggregation:
        Per chunk lmppl returns ``ppl_i = exp(mean_NLL_i)`` over the
        chunk's masked positions. We aggregate as a length-weighted
        log-mean:

            final_ppl = exp(sum_i w_i * log(ppl_i) / sum_i w_i)

        where ``w_i`` is the number of content tokens in chunk ``i``.
        Equivalently, this is ``exp(total_NLL / total_scored_tokens)``,
        the same convention used by :class:`PPLBackend` and
        :class:`MiniconsStridedBackend`. Tokens in overlapping regions
        contribute to multiple chunks.

    Notes:
        - Requires the model to have a *fast* tokenizer (so that
          ``return_offsets_mapping=True`` works). Initialization raises
          ``ValueError`` if only a slow tokenizer is available.
        - When the input document already fits within one chunk, the
          score reduces exactly to what :class:`LmpplBackend` would
          return -- there is no behavior change in the short-input case.

    Example:
        >>> backend = LmpplStridedBackend(
        ...     'microsoft/deberta-v3-small', windows_size=508, stride=256,
        ... )
        >>> ppl = backend.get_perplexity(long_text)
    """

    def __init__(
        self,
        model_name: str,
        windows_size: int = 512,
        stride: Optional[int] = None,
        batch_size: int = 32,
        show_progress: bool = False,
    ):
        """Initialize the strided lmppl backend.

        Args:
            model_name: Hugging Face model name or path for a masked LM.
            windows_size: Total tokenized window length per chunk,
                *including* the two slots lmppl will use for the
                tokenizer's special tokens when re-tokenizing each chunk.
                Forwarded to ``lmppl.MaskedLM`` as its ``max_length``.
                Each chunk thus covers ``windows_size - 2`` content tokens.
            stride: Step in content tokens between consecutive chunks.
                Defaults to ``max(1, (windows_size - 2) // 2)`` (~50%
                overlap). Set to ``windows_size - 2`` for disjoint chunks.
            batch_size: Inference batch size forwarded to lmppl when
                scoring the per-document chunk list.
            show_progress: If True, display a tqdm progress bar over
                documents.
        """
        try:
            import lmppl
        except ImportError:
            raise ImportError(
                "lmppl is required for this backend. Install it with: pip install lmppl"
            )

        if windows_size < 3:
            raise ValueError(
                f"windows_size must be >= 3 (got {windows_size}); two slots "
                "are reserved for the special tokens lmppl adds per chunk."
            )

        # See the note in LmpplBackend about positional vs keyword argument.
        # lmppl's own knob is still called ``max_length``; we just expose it
        # under the ``windows_size`` name for consistency with the other
        # strided backends.
        self.scorer = lmppl.MaskedLM(model_name, max_length=windows_size)
        self.model_name = model_name
        self.windows_size = windows_size
        self.batch_size = batch_size
        self.show_progress = show_progress

        # Load a fast tokenizer for the splitting step. We need
        # offset_mapping, which only fast tokenizers expose. The lmppl
        # scorer holds its own tokenizer instance, but it may or may not be
        # fast depending on what lmppl picked; loading our own keeps the
        # splitting deterministic and decoupled from lmppl internals.
        split_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not getattr(split_tokenizer, "is_fast", False):
            raise ValueError(
                f"LmpplStridedBackend requires a fast tokenizer for "
                f"`{model_name}` (needed for offset_mapping), but only a "
                "slow tokenizer is available. Use LmpplBackend instead, or "
                "score with MiniconsStridedBackend / PPLBackend."
            )
        self._split_tokenizer = split_tokenizer

        content_window = windows_size - 2
        self._content_window = content_window
        if stride is None:
            stride = max(1, content_window // 2)
        if stride <= 0:
            raise ValueError(f"stride must be positive (got {stride}).")
        self.stride = stride

    def _split_text(self, text: str) -> Tuple[List[str], List[int]]:
        """Split a single document into overlapping text chunks.

        Uses the fast tokenizer's offset mapping to find exact character
        positions of token boundaries in the original text, then slices
        substrings of ``self._content_window`` tokens with ``self.stride``
        between chunk starts.

        Returns:
            A pair ``(chunks, lengths)`` where ``chunks`` is a list of
            substrings of ``text`` and ``lengths[i]`` is the number of
            content tokens covered by ``chunks[i]`` (used as the
            aggregation weight). Both lists are empty for an empty input.
        """
        if not text:
            return [], []

        encoded = self._split_tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        offsets = encoded["offset_mapping"]
        n = len(offsets)
        if n == 0:
            return [], []

        chunks: List[str] = []
        lengths: List[int] = []
        start = 0
        while start < n:
            end = min(start + self._content_window, n)
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            chunks.append(text[char_start:char_end])
            lengths.append(end - start)
            if end >= n:
                break  # this chunk already reaches the end of the doc
            start += self.stride

        return chunks, lengths

    def get_perplexity(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        """Compute strided pseudo-perplexity for one or more texts.

        Args:
            text: A single string or list of strings to score.

        Returns:
            A single perplexity score or list of scores. Each per-document
            score is the length-weighted log-mean of per-chunk lmppl
            perplexities, exponentiated.
        """
        single = isinstance(text, str)
        if single:
            text = [text]

        iterator: Union[range, "tqdm"] = range(len(text))
        if self.show_progress and len(text) > 1:
            iterator = tqdm(
                range(len(text)),
                desc="LmpplStridedBackend",
                total=len(text),
            )

        results: List[float] = []
        for i in iterator:
            chunks, lengths = self._split_text(text[i])
            if not chunks:
                results.append(float("nan"))
                continue

            chunk_ppls = self.scorer.get_perplexity(chunks, batch_size=self.batch_size)
            # lmppl returns a list when given a list, even of length 1.
            if not isinstance(chunk_ppls, list):
                chunk_ppls = [chunk_ppls]

            # Length-weighted mean of log-PPL, then exp. For chunk i with
            # n_i content tokens, log(ppl_i) == mean_NLL_i, so this is
            # equivalent to exp(total_NLL / total_scored_tokens).
            total_weight = sum(lengths) or 1
            log_ppl_weighted = sum(math.log(p) * w for p, w in zip(chunk_ppls, lengths))
            results.append(math.exp(log_ppl_weighted / total_weight))

        return results[0] if single else results


class MiniconsBackend:
    """Backend using minicons library for perplexity scoring.

    Supports masked language models (e.g., BERT, RoBERTa).

    Example:
        >>> backend = MiniconsBackend('bert-base-uncased', device='cpu')
        >>> texts = ['The keys to the cabinet are on the table.']
        >>> scores = backend.get_perplexity(texts)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        pll_method: Literal["within_word_l2r", "original"] = "within_word_l2r",
        max_length: Optional[int] = None,
        batch_size: int = 8,
        model_batch_size: int = 32,
        show_progress: bool = False,
    ):
        """Initialize the minicons backend.

        Args:
            model_name: Hugging Face model name or path for a masked LM.
            device: Device to run the model on ('cpu' or 'cuda').
            pll_method: Pseudo-log-likelihood scoring strategy passed to
                minicons (``"original"`` or ``"within_word_l2r"``).
            max_length: Maximum tokenized sequence length (including special
                tokens) to feed into the model. Inputs longer than this are
                truncated with a warning before being passed to minicons,
                which by itself performs no truncation and would otherwise
                crash on models with absolute positional embeddings. When
                ``None``, the limit is auto-detected from the tokenizer and
                model config.
            batch_size: Number of input texts processed per outer minicons
                call. Mostly a throughput/IO knob; per-text-batch chunks
                are still tokenized with their own longest-padding.
            model_batch_size: Number of masked-variant rows fed through the
                model in a single forward pass. This is the *actual* GPU
                memory knob: minicons' built-in ``compute_stats`` runs all
                masked variants of a document (one per non-special token,
                each as a full-length sequence) in one forward, which OOMs
                on long inputs. When ``model_batch_size > 0`` this class
                bypasses ``compute_stats`` and runs the per-document forward
                in chunks of this many rows. Set to ``0`` to fall back to
                minicons' single-forward behavior.
            show_progress: If True, display a tqdm progress bar over
                text-level batches when more than one batch is processed.
        """
        try:
            from minicons import scorer
        except ImportError:
            raise ImportError(
                "minicons is required for this backend. "
                "Install it with: pip install minicons"
            )

        self._scorer = scorer.MaskedLMScorer(model_name, device)
        self.model_name = model_name
        self.device = device
        self.pll_method = pll_method
        self.tokenizer = self._scorer.tokenizer
        self.max_length = self._resolve_max_length(max_length)
        self.batch_size = batch_size
        self.model_batch_size = model_batch_size
        self.show_progress = show_progress

    def _resolve_max_length(self, max_length: Optional[int]) -> int:
        """Determine an effective max sequence length for the underlying model.

        Thin instance-method wrapper around :func:`_resolve_model_max_length`.
        """
        return _resolve_model_max_length(self.tokenizer, self._scorer.model, max_length)

    def get_perplexity(
        self, text: Union[str, List[str]], reduction=None
    ) -> Union[float, List[float]]:
        """Compute pseudo-perplexity for one or more texts.

        By default this returns ``exp(mean negative log-probability per
        token)``, i.e. the pseudo-perplexity of the sequence under the masked
        LM scoring scheme selected by ``pll_method``.

        Args:
            text: A single string or list of strings to score.
            reduction: Optional custom reduction function applied to the
                per-token log-probability tensor returned by minicons. If
                provided, its output is returned as-is (no exponentiation),
                which is useful for retrieving raw log-likelihoods or
                custom aggregates.

        Returns:
            A single perplexity score or list of scores when ``reduction``
            is ``None``; otherwise whatever the supplied ``reduction``
            produces.
        """
        return_perplexity = reduction is None
        if return_perplexity:
            reduction = _mean_negative_log_prob

        single = isinstance(text, str)
        if single:
            text = [text]

        # Determine the effective text-level batch size. A non-positive
        # value or a value at least as large as the input disables chunking.
        bs = self.batch_size if self.batch_size and self.batch_size > 0 else len(text)
        bs = min(bs, len(text)) if len(text) > 0 else bs

        scores: List[float] = []
        batch_starts = range(0, len(text), bs) if bs > 0 else range(0)

        iterator = batch_starts
        if self.show_progress and len(text) > bs:
            iterator = tqdm(
                batch_starts,
                total=len(batch_starts),
                desc="MiniconsBackend",
            )

        use_chunked_forward = self.model_batch_size and self.model_batch_size > 0

        for start in iterator:
            chunk = text[start : start + bs]
            encoded = self._encode_with_truncation(chunk)

            if use_chunked_forward:
                # Bypass minicons' compute_stats: iterate the per-document
                # masked-variant generator and run each document's forward
                # in row-chunks of self.model_batch_size to bound memory.
                masked_iter = self._scorer.prepare_text(
                    encoded, PLL_metric=self.pll_method
                )
                for masked_tensors in masked_iter:
                    doc_logprobs = self._score_one_doc(masked_tensors)
                    scores.append(reduction(doc_logprobs))
            else:
                # Original path: minicons runs ALL masked variants of ALL
                # docs in this chunk in a single model forward.
                chunk_scores = self._scorer.sequence_score(
                    encoded, reduction=reduction, PLL_metric=self.pll_method
                )
                scores.extend(chunk_scores)

        if return_perplexity:
            scores = [math.exp(s) for s in scores]

        return scores[0] if single else scores

    def _score_one_doc(self, masked_tensors) -> torch.Tensor:
        """Run a single document's masked-LM forward in row-sized chunks.

        Thin instance-method wrapper around
        :func:`_minicons_chunked_forward` so the chunked-forward logic can
        be shared with :class:`MiniconsStridedBackend`.
        """
        return _minicons_chunked_forward(
            self._scorer, masked_tensors, self.model_batch_size
        )

    def _encode_with_truncation(self, texts: List[str]):
        """Tokenize ``texts``, truncating to ``self.max_length`` if needed.

        Minicons' own tokenization path (``LMScorer.encode``) does not pass
        ``truncation=True`` or any ``max_length`` to the tokenizer, so long
        inputs are forwarded whole to the model and typically crash with an
        ``index out of range`` error on models with learned absolute
        positional embeddings. This method enforces the limit and warns
        when truncation is applied.

        Returns a ``BatchEncoding`` accepted directly by
        ``MaskedLMScorer.prepare_text``.
        """
        # First, measure untruncated lengths so we can report which inputs
        # would be silently cut. Tokenization without a model forward is
        # cheap relative to the masked-LM scoring that follows.
        raw = self.tokenizer(
            texts,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        over = [
            (i, len(ids))
            for i, ids in enumerate(raw["input_ids"])
            if len(ids) > self.max_length
        ]
        if over:
            preview = ", ".join(f"#{i} ({n} tokens)" for i, n in over[:5])
            more = "" if len(over) <= 5 else f" (and {len(over) - 5} more)"
            warnings.warn(
                f"MiniconsBackend: {len(over)}/{len(texts)} input(s) exceed "
                f"max_length={self.max_length} and will be truncated: "
                f"{preview}{more}.",
                stacklevel=3,
            )

        # Re-tokenize with truncation and longest-padding so minicons can
        # iterate over a rectangular BatchEncoding (matching the shape it
        # would normally produce internally, just length-bounded).
        encoded = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        return encoded


def _mean_negative_log_prob(x):
    """Default reduction: mean negative token log-probability (mean NLL)."""
    return -x.mean(0).item()


def _negative_log_prob_sum(x):
    """Reduction returning the negative sum of token log-probabilities."""
    return -x.sum(0).item()


def _resolve_model_max_length(tokenizer, model, max_length: Optional[int]) -> int:
    """Determine an effective max sequence length for a HF model+tokenizer.

    Prefers an explicit user value, then ``tokenizer.model_max_length``
    (ignoring HF's huge sentinel value), then common model-config attributes
    (``max_position_embeddings`` / ``n_positions`` / ``max_seq_length``),
    and finally falls back to 512.
    """
    if max_length is not None:
        return int(max_length)

    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max is not None and tok_max < _HF_MAX_LENGTH_SENTINEL:
        return int(tok_max)

    config = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "max_seq_length"):
        value = getattr(config, attr, None)
        if value is not None and value < _HF_MAX_LENGTH_SENTINEL:
            return int(value)

    return 512


def _minicons_chunked_forward(
    scorer, masked_tensors, model_batch_size: int
) -> torch.Tensor:
    """Run a single document's masked-LM forward in row-sized chunks.

    Mirrors the scoring math of ``minicons.MaskedLMScorer.compute_stats``
    but breaks the forward into row-batches of ``model_batch_size`` to bound
    GPU memory. Used by both :class:`MiniconsBackend` and
    :class:`MiniconsStridedBackend`.

    Args:
        scorer: A minicons ``MaskedLMScorer`` holding the model and device.
        masked_tensors: One tuple yielded by ``scorer.get_masked_tensors``,
            i.e. ``(token_ids_masked, attention_mask, target_token_ids,
            target_token_indices)`` for a single document.
        model_batch_size: Number of masked-variant rows per model forward.

    Returns:
        1D CPU tensor of per-target log-probabilities, in the same order
        minicons' ``compute_stats`` would have produced for the document.
    """
    token_ids, attn_mask, target_token_ids, target_token_indices = masked_tensors
    n_rows = token_ids.shape[0]
    if n_rows == 0:
        return torch.empty(0)

    device = scorer.model.device
    sub_bs = max(1, int(model_batch_size))

    out_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, n_rows, sub_bs):
            ti = token_ids[i : i + sub_bs].to(device)
            am = attn_mask[i : i + sub_bs].to(device)
            indices = target_token_indices[i : i + sub_bs].to(device)
            tids = target_token_ids[i : i + sub_bs].to(device)

            logits = scorer.model(ti, attention_mask=am).logits  # (B, T, V)

            row = torch.arange(logits.shape[0], device=device)
            # Pick the logits at each masked target position: (B, V)
            target_logits = logits[row, indices]
            # Log-softmax over the vocabulary, then gather the score of
            # the correct token id for each row.
            log_probs = target_logits - target_logits.logsumexp(-1, keepdim=True)
            chunk_lp = log_probs[row, tids]  # (B,)
            out_chunks.append(chunk_lp.detach().cpu())

            # Drop large intermediates before the next chunk so the
            # allocator can reuse the memory.
            del logits, target_logits, log_probs, chunk_lp

    return torch.cat(out_chunks)


class MiniconsStridedBackend:
    """Minicons-based backend with sliding-window scoring over full documents.

    Where :class:`MiniconsBackend` truncates each input to the model's max
    length (matching how minicons itself behaves under the hood), this
    backend scores the entire document by:

    1. Tokenizing the full document with ``add_special_tokens=False`` and
       ``return_offsets_mapping=True`` to locate exact character boundaries
       of each token in the original text.
    2. Sliding a content window of ``windows_size − n_special`` tokens
       (where ``n_special`` is the number of framing tokens the tokenizer
       adds, typically 2 for ``[CLS]``/``[SEP]``) across the token sequence.
    3. For each window, extracting the corresponding **text substring** and
       re-tokenizing it with ``add_special_tokens=True``. The resulting
       ``BatchEncoding`` has proper ``_encodings`` so
       ``word_ids()`` works — enabling both ``"original"`` and
       ``"within_word_l2r"`` PLL metrics.
    4. Scoring each window via minicons' ``prepare_text`` + our
       chunked-forward helper, concatenating per-token log-probabilities
       across windows, and returning ``exp(mean NLL)``.

    Notes:
        - Requires a *fast* tokenizer (offset-mapping support).
          Initialization raises ``ValueError`` if only a slow tokenizer is
          available.
        - Because each window is independently re-tokenized from its text
          substring, BPE/SentencePiece merges at window edges may differ
          very slightly from the full-document tokenization. The effect on
          per-token NLLs is negligible (the model sees a valid, naturally
          framed input for each window).
        - With overlapping windows (``stride < content_window``) tokens
          near boundaries are scored more than once — the aggregation
          treats every ``(token, window)`` pair as a sample, which is the
          same convention used by :class:`PPLBackend`.

    Example:
        >>> backend = MiniconsStridedBackend(
        ...     'bert-base-uncased', device='cuda',
        ...     windows_size=512, stride=256, model_batch_size=8,
        ...     pll_method='within_word_l2r',
        ... )
        >>> ppl = backend.get_perplexity(long_text)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        windows_size: int = 512,
        stride: int = 256,
        model_batch_size: int = 32,
        show_progress: bool = False,
        pll_method: Literal["within_word_l2r", "original"] = "original",
    ):
        """Initialize the strided minicons backend.

        Args:
            model_name: Hugging Face model name or path for a masked LM.
            device: Device to run the model on ('cpu' or 'cuda').
            windows_size: Total tokenized window length, *including*
                special tokens. Must be ``>= 3`` and is automatically
                capped to the model's max sequence length.
            stride: Step (in content tokens) between successive windows.
                ``stride == windows_size - n_special`` gives disjoint
                windows; smaller values produce overlap.
            model_batch_size: Masked-variant rows per model forward, the
                same memory knob as on :class:`MiniconsBackend`.
            show_progress: If True, display a tqdm progress bar over
                documents.
            pll_method: PLL scoring strategy (``"original"`` or
                ``"within_word_l2r"``). Both work because each window is
                re-tokenized with the fast tokenizer, producing a proper
                ``BatchEncoding`` with ``word_ids()`` support.
        """
        try:
            from minicons import scorer
        except ImportError:
            raise ImportError(
                "minicons is required for this backend. "
                "Install it with: pip install minicons"
            )

        self._scorer = scorer.MaskedLMScorer(model_name, device)
        self.model_name = model_name
        self.device = device
        self.tokenizer = self._scorer.tokenizer
        self.model_batch_size = model_batch_size
        self.show_progress = show_progress
        self.pll_method = pll_method

        # Fast-tokenizer check: we need offset_mapping for text-based
        # window construction, and _encodings for word_ids().
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError(
                f"MiniconsStridedBackend requires a fast tokenizer for "
                f"`{model_name}` (needed for offset_mapping and "
                f"word_ids support). Only a slow tokenizer is available."
            )

        # How many framing tokens does the tokenizer add for a single
        # sentence? (e.g. 2 for BERT/RoBERTa: [CLS] + [SEP]).
        n_special = self.tokenizer.num_special_tokens_to_add(pair=False)

        if windows_size < n_special + 1:
            raise ValueError(
                f"windows_size must be >= {n_special + 1} (got {windows_size}); "
                f"{n_special} slot(s) are reserved for the tokenizer's "
                "special tokens."
            )
        if stride <= 0:
            raise ValueError(f"stride must be positive (got {stride}).")

        # Cap windows_size at the model's actual max length.
        model_max = _resolve_model_max_length(self.tokenizer, self._scorer.model, None)
        if windows_size > model_max:
            warnings.warn(
                f"MiniconsStridedBackend: windows_size={windows_size} exceeds "
                f"the model's max sequence length ({model_max}); capping.",
                stacklevel=2,
            )
            windows_size = model_max

        self.windows_size = windows_size
        self._content_window = windows_size - n_special
        self.stride = stride

    def get_perplexity(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        """Compute strided pseudo-perplexity for one or more texts.

        Args:
            text: A single string or list of strings to score.

        Returns:
            A single perplexity score or list of scores. Each per-document
            score is ``exp(mean NLL across all (token, window) pairs)``.
        """
        single = isinstance(text, str)
        if single:
            text = [text]

        iterator: Union[range, "tqdm"] = range(len(text))
        if self.show_progress and len(text) > 1:
            iterator = tqdm(
                range(len(text)),
                desc="MiniconsStridedBackend",
                total=len(text),
            )

        scores: List[float] = []

        for i in iterator:
            text_i = text[i]

            # Tokenize the full doc without specials to get offset_mapping.
            full = self.tokenizer(
                text_i,
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
            offsets = full["offset_mapping"]
            n = len(offsets)
            if n == 0:
                scores.append(float("nan"))
                continue

            # Slide a window of _content_window tokens across the document,
            # extract the text substring for each, and re-tokenize it with
            # specials so the BatchEncoding has proper _encodings
            # (word_ids() support).
            doc_logprobs: List[torch.Tensor] = []
            start = 0
            while start < n:
                end = min(start + self._content_window, n)
                char_start = offsets[start][0]
                char_end = offsets[end - 1][1]
                chunk_text = text_i[char_start:char_end]

                # Tokenize the window text *with* specials → proper
                # BatchEncoding whose _encodings track word boundaries.
                window_enc = self.tokenizer(
                    [chunk_text],
                    padding=False,
                    truncation=True,
                    max_length=self.windows_size,
                    return_attention_mask=True,
                )

                for masked_tensors in self._scorer.prepare_text(
                    window_enc, PLL_metric=self.pll_method
                ):
                    window_lp = _minicons_chunked_forward(
                        self._scorer, masked_tensors, self.model_batch_size
                    )
                    doc_logprobs.append(window_lp)

                if start + self._content_window >= n:
                    break
                start += self.stride

            if not doc_logprobs:
                scores.append(float("nan"))
                continue

            all_lp = torch.cat(doc_logprobs)
            mean_nll = -all_lp.mean(0).item()
            scores.append(math.exp(mean_nll))

        return scores[0] if single else scores


def get_backend(
    backend: str, model_name: str, device: str = "cpu", **kwargs
) -> Union[
    LmpplBackend,
    LmpplStridedBackend,
    MiniconsBackend,
    MiniconsStridedBackend,
    PPLBackend,
]:
    """Factory function to create a backend instance.

    Args:
        backend: Backend name, one of 'lmppl', 'lmppl_strided', 'minicons',
            'minicons_strided', or 'pppl'.
        model_name: Hugging Face model name or path.
        device: Device to run the model on (used for 'minicons',
            'minicons_strided', and 'pppl').
        **kwargs: Additional keyword arguments passed to the backend.

    Returns:
        An instance of the requested backend.

    Raises:
        ValueError: If the backend name is not recognized.
    """
    if backend == "lmppl":
        return LmpplBackend(model_name, **kwargs)
    elif backend == "lmppl_strided":
        return LmpplStridedBackend(model_name, **kwargs)
    elif backend == "minicons":
        return MiniconsBackend(model_name, device=device, **kwargs)
    elif backend == "minicons_strided":
        return MiniconsStridedBackend(model_name, device=device, **kwargs)
    elif backend == "pppl":
        return PPLBackend(model_name, device=device, **kwargs)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from 'lmppl', "
            "'lmppl_strided', 'minicons', 'minicons_strided', or 'pppl'."
        )
