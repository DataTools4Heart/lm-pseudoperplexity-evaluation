"""
Backend wrappers for perplexity scoring using lmppl and minicons libraries.

This module provides a unified interface for computing perplexity scores
using different backends:
- lmppl: https://github.com/asahi417/lmppl
- minicons: https://github.com/kanishkamisra/minicons
"""

from typing import List, Union


class LmpplBackend:
    """Backend using lmppl library for perplexity scoring.

    Supports masked language models (e.g., DeBERTa, BERT, RoBERTa).

    Example:
        >>> backend = LmpplBackend('microsoft/deberta-v3-small')
        >>> texts = ['I am happy.', 'I am sad.']
        >>> scores = backend.get_perplexity(texts)
    """

    def __init__(self, model_name: str):
        """Initialize the lmppl backend.

        Args:
            model_name: Hugging Face model name or path for a masked LM.
        """
        try:
            import lmppl
        except ImportError:
            raise ImportError(
                "lmppl is required for this backend. "
                "Install it with: pip install lmppl"
            )

        self.scorer = lmppl.MaskedLM(model_name)
        self.model_name = model_name

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
        return self.scorer.get_perplexity(text)


class MiniconsBackend:
    """Backend using minicons library for perplexity scoring.

    Supports masked language models (e.g., BERT, RoBERTa).

    Example:
        >>> backend = MiniconsBackend('bert-base-uncased', device='cpu')
        >>> texts = ['The keys to the cabinet are on the table.']
        >>> scores = backend.get_perplexity(texts)
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        """Initialize the minicons backend.

        Args:
            model_name: Hugging Face model name or path for a masked LM.
            device: Device to run the model on ('cpu' or 'cuda').
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

    def get_perplexity(
        self, text: Union[str, List[str]], reduction=None
    ) -> Union[float, List[float]]:
        """Compute perplexity-like scores for one or more texts.

        Uses the negative sum of log-probabilities as the score by default,
        which corresponds to the pseudo-perplexity of the sequence.

        Args:
            text: A single string or list of strings to score.
            reduction: Optional custom reduction function. If None, uses
                the negative sum of token log-probabilities.

        Returns:
            A single score or list of scores.
        """
        if reduction is None:
            reduction = _negative_log_prob_sum

        if isinstance(text, str):
            text = [text]
            return self._scorer.sequence_score(text, reduction=reduction)[0]
        return self._scorer.sequence_score(text, reduction=reduction)


def _negative_log_prob_sum(x):
    """Default reduction: negative sum of token log-probabilities."""
    return -x.sum(0).item()


def get_backend(
    backend: str, model_name: str, device: str = "cpu", **kwargs
) -> Union[LmpplBackend, MiniconsBackend]:
    """Factory function to create a backend instance.

    Args:
        backend: Backend name, either 'lmppl' or 'minicons'.
        model_name: Hugging Face model name or path.
        device: Device to run the model on (only used for minicons).
        **kwargs: Additional keyword arguments passed to the backend.

    Returns:
        An instance of the requested backend.

    Raises:
        ValueError: If the backend name is not recognized.
    """
    if backend == "lmppl":
        return LmpplBackend(model_name, **kwargs)
    elif backend == "minicons":
        return MiniconsBackend(model_name, device=device, **kwargs)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from 'lmppl' or 'minicons'."
        )
