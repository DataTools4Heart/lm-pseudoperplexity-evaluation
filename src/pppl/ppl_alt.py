import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

try:
    from pppl.config import (
        DEFAULT_MODELS,
        ERROR_MESSAGES,
        MODEL_SETTINGS,
        PROCESSING_SETTINGS,
        UI_SETTINGS,
        VIZ_SETTINGS,
    )
except ImportError:
    # Fallback configuration if config.py is not available
    DEFAULT_MODELS = {
        "decoder": ["gpt2", "distilgpt2"],
        "encoder": ["bert-base-uncased", "distilbert-base-uncased"],
    }
    MODEL_SETTINGS = {"max_length": 512}
    VIZ_SETTINGS = {
        "max_perplexity_display": 5000.0,
        "color_scheme": {
            "low_perplexity": {"r": 46, "g": 204, "b": 113},
            "medium_perplexity": {"r": 241, "g": 196, "b": 15},
            "high_perplexity": {"r": 231, "g": 76, "b": 60},
            "background_alpha": 0.7,
            "border_alpha": 0.9,
        },
        "thresholds": {"low_threshold": 0.3, "high_threshold": 0.7},
        "displacy_options": {"ents": ["PP"], "colors": {}},
    }
    PROCESSING_SETTINGS = {
        "epsilon": 1e-10,
        "default_mask_probability": 0.15,
        "min_mask_probability": 0.05,
        "max_mask_probability": 0.5,
        "default_min_samples": 10,
        "min_samples_range": (5, 50),
    }
    UI_SETTINGS = {
        "title": "📈 Perplexity Viewer",
        "description": "Visualize per-token perplexity using color gradients.",
        "examples": [
            {
                "text": "The quick brown fox jumps over the lazy dog.",
                "model": "gpt2",
                "type": "decoder",
                "mask_prob": 0.15,
                "min_samples": 10,
            },
            {
                "text": "The capital of France is Paris.",
                "model": "bert-base-uncased",
                "type": "encoder",
                "mask_prob": 0.15,
                "min_samples": 10,
            },
            {
                "text": "Quantum entanglement defies classical physics intuition completely.",
                "model": "distilgpt2",
                "type": "decoder",
                "mask_prob": 0.15,
                "min_samples": 10,
            },
            {
                "text": "Machine learning requires large datasets for training.",
                "model": "distilbert-base-uncased",
                "type": "encoder",
                "mask_prob": 0.2,
                "min_samples": 15,
            },
            {
                "text": "Artificial intelligence transforms modern computing paradigms.",
                "model": "bert-base-uncased",
                "type": "encoder",
                "mask_prob": 0.1,
                "min_samples": 20,
            },
        ],
    }
    ERROR_MESSAGES = {
        "empty_text": "Please enter some text to analyze.",
        "model_load_error": "Error loading model {model_name}: {error}",
        "processing_error": "Error processing text: {error}",
    }
warnings.filterwarnings("ignore")

# Global variables to cache models
cached_models = {}
cached_tokenizers = {}


def is_special_character(token):
    """
    Check if a token is only special characters/punctuation.

    Args:
        token: The token string to check

    Returns:
        True if token contains only special characters, False otherwise

    Examples:
        >>> is_special_character(".")
        True
        >>> is_special_character(",")
        True
        >>> is_special_character("hello")
        False
        >>> is_special_character("Ġ,")
        True
        >>> is_special_character("##!")
        True
    """
    # Clean up common tokenizer artifacts
    clean_token = (
        token.replace("</w>", "")
        .replace("##", "")
        .replace("Ġ", "")
        .replace("Ċ", "")
        .strip()
    )

    # Check if empty after cleaning
    if not clean_token:
        return True

    # Check if token contains only punctuation and special characters
    return all(not c.isalnum() for c in clean_token)


def load_model_and_tokenizer(model_name, model_type):
    """Load and cache model and tokenizer"""
    cache_key = f"{model_name}_{model_type}"

    if cache_key not in cached_models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if model_type == "decoder":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                    if torch.cuda.is_available()
                    else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )
            else:  # encoder
                model = AutoModelForMaskedLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                    if torch.cuda.is_available()
                    else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )

            model.eval()  # Set to evaluation mode
            cached_models[cache_key] = model
            cached_tokenizers[cache_key] = tokenizer

            return model, tokenizer
        except Exception as e:
            raise gr.Error(
                ERROR_MESSAGES["model_load_error"].format(
                    model_name=model_name, error=str(e)
                )
            )

    return cached_models[cache_key], cached_tokenizers[cache_key]


def calculate_decoder_perplexity(text, model, tokenizer):
    """Calculate perplexity for decoder models (like GPT)"""
    device = next(model.parameters()).device

    # Tokenize the text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MODEL_SETTINGS["max_length"],
    )
    input_ids = inputs.input_ids.to(device)

    if input_ids.size(1) < 2:
        raise Exception("Text is too short for perplexity calculation.")

    # Calculate overall perplexity
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    # Get token-level perplexities
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Calculate per-token losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        token_perplexities = torch.exp(token_losses).cpu().numpy()

        # Get tokens (excluding the first one since we predict next tokens)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0][1:])

        # Clean up tokens for display and filter special characters
        cleaned_tokens = []
        filtered_perplexities = []
        for token, token_perp in zip(tokens, token_perplexities):
            # Skip special characters
            if is_special_character(token):
                continue

            if token.startswith("Ġ"):
                cleaned_tokens.append(token[1:])  # Remove Ġ prefix
            elif token.startswith("Ċ"):
                cleaned_tokens.append(token[1:])
            elif token.startswith("##"):
                cleaned_tokens.append(token[2:])  # Remove ## prefix
            else:
                cleaned_tokens.append(token)
            filtered_perplexities.append(token_perp)

    return perplexity, cleaned_tokens, np.array(filtered_perplexities)


def calculate_encoder_perplexity(
    text, model, tokenizer, mask_probability=0.15, min_samples_per_token=10
):
    """Calculate pseudo-perplexity for encoder models using statistical sampling with multiple token masking"""
    device = next(model.parameters()).device

    # Tokenize the text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MODEL_SETTINGS["max_length"],
    )
    input_ids = inputs.input_ids.to(device)

    if input_ids.size(1) < 3:  # Need at least [CLS] + 1 token + [SEP]
        raise Exception("Text is too short for MLM perplexity calculation.")

    seq_length = input_ids.size(1)
    special_token_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    }

    # Get content token indices (excluding special tokens)
    content_token_indices = [
        i for i in range(seq_length) if input_ids[0, i].item() not in special_token_ids
    ]

    if not content_token_indices:
        raise Exception("No content tokens found for analysis.")

    # Initialize storage for per-token perplexity samples
    token_perplexity_samples = {idx: [] for idx in content_token_indices}

    # Calculate overall average perplexity and collect samples
    all_losses = []
    max_iterations = (
        min_samples_per_token * 50
    )  # Safety limit to prevent infinite loops
    iteration = 0

    with torch.no_grad():
        while iteration < max_iterations:
            # Create a copy for masking
            masked_input = input_ids.clone()
            masked_indices = []

            # Randomly mask tokens based on mask_probability
            for idx in content_token_indices:
                if torch.rand(1).item() < mask_probability:
                    masked_indices.append(idx)
                    masked_input[0, idx] = tokenizer.mask_token_id

            # Skip if no tokens were masked
            if not masked_indices:
                iteration += 1
                continue

            # Get model predictions
            outputs = model(masked_input)
            predictions = outputs.logits

            # Calculate perplexity for each masked token
            for idx in masked_indices:
                original_token_id = input_ids[0, idx]
                pred_scores = predictions[0, idx]
                prob = F.softmax(pred_scores, dim=-1)[original_token_id]
                loss = -torch.log(prob + PROCESSING_SETTINGS["epsilon"])
                perplexity = math.exp(loss.item())

                # Store sample for this token
                token_perplexity_samples[idx].append(perplexity)
                all_losses.append(loss.item())

            iteration += 1

            # Check if we have enough samples for all tokens
            min_samples_collected = min(
                len(samples) for samples in token_perplexity_samples.values()
            )
            if min_samples_collected >= min_samples_per_token:
                break

    # Calculate overall average perplexity
    if all_losses:
        avg_loss = np.mean(all_losses)
        overall_perplexity = math.exp(avg_loss)
    else:
        overall_perplexity = float("inf")

    # Calculate mean perplexity per token for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_perplexities = []

    for i in range(len(tokens)):
        if input_ids[0, i].item() in special_token_ids:
            token_perplexities.append(1.0)  # Low perplexity for special tokens
        elif i in token_perplexity_samples and token_perplexity_samples[i]:
            # Use mean of collected samples
            token_perplexities.append(np.mean(token_perplexity_samples[i]))
        else:
            # Fallback if no samples collected (shouldn't happen with proper min_samples)
            token_perplexities.append(2.0)

    # Clean up tokens for display and filter special characters
    cleaned_tokens = []
    filtered_perplexities = []
    for idx, (token, token_perp) in enumerate(zip(tokens, token_perplexities)):
        # Skip special characters and tokenizer special tokens
        if input_ids[0, idx].item() in special_token_ids:
            continue
        if is_special_character(token):
            continue

        if token.startswith("##"):
            cleaned_tokens.append(token[2:])
        elif token.startswith("Ċ"):
            cleaned_tokens.append(token[1:])
        elif token.startswith("Ġ"):
            cleaned_tokens.append(token[1:])
        else:
            cleaned_tokens.append(token)
        filtered_perplexities.append(token_perp)

    return overall_perplexity, cleaned_tokens, np.array(filtered_perplexities)


def process_text(text, model_name, model_type, mask_probability=0.15, min_samples=10):
    """Main processing function"""
    if not text.strip():
        return ERROR_MESSAGES["empty_text"], "", pd.DataFrame()

    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_name, model_type)

        # Calculate perplexity
        if model_type == "decoder":
            avg_perplexity, tokens, token_perplexities = calculate_decoder_perplexity(
                text, model, tokenizer
            )
            sampling_info = ""
        else:  # encoder
            avg_perplexity, tokens, token_perplexities = calculate_encoder_perplexity(
                text, model, tokenizer, mask_probability, min_samples
            )
            sampling_info = f"**Mask Probability:** {mask_probability:.1%}  \n**Min Samples per Token:** {min_samples}  \n"

        # Create summary
        summary = f"""
### Analysis Results

**Model:** `{model_name}`
**Model Type:** {model_type.title()}
**Average Perplexity:** {avg_perplexity:.4f}
**Number of Tokens:** {len(tokens)}
{sampling_info}"""

        return summary, list(zip(tokens, token_perplexities))

    except Exception as e:
        error_msg = ERROR_MESSAGES["processing_error"].format(error=str(e))
        return error_msg, "", []
