import math
import warnings

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

try:
    from config import (
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
        raise gr.Error("Text is too short for perplexity calculation.")

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
        raise gr.Error("Text is too short for MLM perplexity calculation.")

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
        raise gr.Error("No content tokens found for analysis.")

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
        else:
            cleaned_tokens.append(token)
        filtered_perplexities.append(token_perp)

    return overall_perplexity, cleaned_tokens, np.array(filtered_perplexities)


def perplexity_to_color(perplexity, min_perp=1, max_perp=1000):
    """
    Convert perplexity to a color on a gradient from green to red.
    Uses logarithmic scale for better visual distribution.

    Args:
        perplexity: The perplexity value
        min_perp: Minimum perplexity (maps to green)
        max_perp: Maximum perplexity (maps to red)

    Returns:
        Tuple of (r, g, b) values as integers (0-255)
    """
    # Clamp perplexity to range
    perp = max(min_perp, min(max_perp, perplexity))

    # Use logarithmic scale for better distribution
    log_min = math.log(min_perp)
    log_max = math.log(max_perp)
    log_perp = math.log(perp)

    # Normalize to 0-1 range
    normalized = (log_perp - log_min) / (log_max - log_min)

    # Create color gradient from green to red via yellow
    # Green: (0, 178, 0) - HSL(120, 100%, 35%)
    # Yellow: (255, 255, 0) - HSL(60, 100%, 50%)
    # Red: (255, 0, 0) - HSL(0, 100%, 50%)

    if normalized < 0.5:
        # Green to Yellow
        factor = normalized * 2  # 0 to 1
        r = int(0 + factor * 255)
        g = int(178 + factor * (255 - 178))
        b = 0
    else:
        # Yellow to Red
        factor = (normalized - 0.5) * 2  # 0 to 1
        r = 255
        g = int(255 * (1 - factor))
        b = 0

    return (r, g, b)


def create_visualization(tokens, perplexities):
    """Create custom HTML visualization with color-coded perplexities"""
    if len(tokens) == 0:
        return "<p>No tokens to visualize.</p>"

    # Cap perplexities for better visualization
    max_perplexity = np.max(perplexities)

    # Normalize perplexities to 0-1 range for color mapping
    normalized_perplexities = np.clip(perplexities / max_perplexity, 0, 1)

    # Create HTML with inline styles for color coding
    html_parts = [
        '<div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.8; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background-color: #fafafa;">',
        '<h3 style="margin-top: 0; color: #333;">Per-token Perplexity Visualization</h3>',
        '<div style="margin-bottom: 15px;">',
        '<span style="font-size: 12px; color: #666;">',
        "🟢 Low perplexity (confident) → 🟡 Medium → 🔴 High perplexity (uncertain)",
        "</span>",
        "</div>",
        '<div style="line-height: 2.0;">',
    ]

    for i, (token, perp, norm_perp) in enumerate(
        zip(tokens, perplexities, normalized_perplexities)
    ):
        # Skip empty tokens
        if not token.strip():
            continue

        # Skip special characters (already filtered in calculation functions)
        if is_special_character(token):
            continue

        # Clean token for display
        # </w>, ##, Ġ, Ċ
        clean_token = (
            token.replace("</w>", "")
            .replace("##", "")
            .replace("Ġ", "")
            .replace("Ċ", "")
            .strip()
        )
        if not clean_token:
            continue

        # Add space before token if needed
        if i > 0 and clean_token[0] not in ".,!?;:":
            html_parts.append(" ")

        # Get color thresholds from configuration
        # low_thresh = VIZ_SETTINGS.get("thresholds", {}).get("low_threshold", 0.3)
        # high_thresh = VIZ_SETTINGS.get("thresholds", {}).get("high_threshold", 0.7)

        # Get colors from configuration
        # low_color = VIZ_SETTINGS["color_scheme"]["low_perplexity"]
        # med_color = VIZ_SETTINGS["color_scheme"]["medium_perplexity"]
        # high_color = VIZ_SETTINGS["color_scheme"]["high_perplexity"]

        # # Map perplexity to color using configuration
        # if norm_perp < low_thresh:  # Low perplexity - green
        #     # Interpolate between green and yellow
        #     factor = norm_perp / low_thresh
        #     red = int(low_color["r"] + factor * (med_color["r"] - low_color["r"]))
        #     green = int(low_color["g"] + factor * (med_color["g"] - low_color["g"]))
        #     blue = int(low_color["b"] + factor * (med_color["b"] - low_color["b"]))
        # elif norm_perp < high_thresh:  # Medium perplexity - yellow/orange
        #     # Interpolate between yellow and red
        #     factor = (norm_perp - low_thresh) / (high_thresh - low_thresh)
        #     red = int(med_color["r"] + factor * (high_color["r"] - med_color["r"]))
        #     green = int(med_color["g"] + factor * (high_color["g"] - med_color["g"]))
        #     blue = int(med_color["b"] + factor * (high_color["b"] - med_color["b"]))
        # else:  # High perplexity - red
        #     # Use high perplexity color, potentially darker for very high values
        #     factor = min((norm_perp - high_thresh) / (1.0 - high_thresh), 1.0)
        #     darken = 0.8 - (factor * 0.3)  # Darken by up to 30%
        #     red = int(high_color["r"] * darken)
        #     green = int(high_color["g"] * darken)
        #     blue = int(high_color["b"] * darken)

        tooltip_text = f"Perplexity: {perp:.3f} (normalized: {norm_perp:.3f})"

        # Clamp values
        # red = max(0, min(255, red))
        # green = max(0, min(255, green))
        # blue = max(0, min(255, blue))

        # Get alpha values from configuration
        bg_alpha = VIZ_SETTINGS["color_scheme"].get("background_alpha", 0.7)
        border_alpha = VIZ_SETTINGS["color_scheme"].get("border_alpha", 0.9)

        # Get RGB color from perplexity
        r, g, b = perplexity_to_color(
            perp, min_perp=1, max_perp=VIZ_SETTINGS["max_perplexity_display"]
        )

        # Create colored span with tooltip
        html_parts.append(
            f'<span style="'
            f"background-color: rgba({r}, {g}, {b}, {bg_alpha}); "
            f"color: #000; "
            f"padding: 2px 4px; "
            f"margin: 1px; "
            f"border-radius: 3px; "
            f"border: 1px solid rgba({r}, {g}, {b}, {border_alpha}); "
            f"font-weight: 500; "
            f"cursor: help; "
            f"display: inline-block;"
            f'" title="{tooltip_text}">{clean_token}</span>'
        )

    html_parts.extend(
        [
            "</div>",
            '<div style="margin-top: 15px; font-size: 12px; color: #666;">',
            f"Max perplexity in visualization: {max_perplexity:.2f} | ",
            f"Total tokens: {len(tokens)}",
            "</div>",
            "</div>",
        ]
    )

    return "".join(html_parts)


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

        # Create visualization
        viz_html = create_visualization(tokens, token_perplexities)

        # Create summary
        summary = f"""
### Analysis Results

**Model:** `{model_name}`
**Model Type:** {model_type.title()}
**Average Perplexity:** {avg_perplexity:.4f}
**Number of Tokens:** {len(tokens)}
{sampling_info}"""

        # Create detailed results table
        df = pd.DataFrame(
            {"Token": tokens, "Perplexity": [f"{p:.4f}" for p in token_perplexities]}
        )

        return summary, viz_html, df

    except Exception as e:
        error_msg = ERROR_MESSAGES["processing_error"].format(error=str(e))
        return error_msg, "", pd.DataFrame()


# Create Gradio interface
with gr.Blocks(title=UI_SETTINGS["title"], theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {UI_SETTINGS['title']}")
    gr.Markdown(UI_SETTINGS["description"])

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter the text you want to analyze...",
                lines=6,
                max_lines=10,
            )

            with gr.Row():
                model_name = gr.Dropdown(
                    label="Model Name",
                    choices=DEFAULT_MODELS["decoder"] + DEFAULT_MODELS["encoder"],
                    value="gpt2",
                    allow_custom_value=True,
                    info="Select a model or enter a custom HuggingFace model name",
                )

                model_type = gr.Radio(
                    label="Model Type",
                    choices=["decoder", "encoder"],
                    value="decoder",
                    info="Decoder for causal LM, Encoder for masked LM",
                )

            # Advanced settings for encoder models
            with gr.Row():
                mask_probability = gr.Slider(
                    label="Mask Probability",
                    minimum=PROCESSING_SETTINGS["min_mask_probability"],
                    maximum=PROCESSING_SETTINGS["max_mask_probability"],
                    value=PROCESSING_SETTINGS["default_mask_probability"],
                    step=0.05,
                    visible=False,
                    info="Probability of masking each token per iteration (encoder only)",
                )

                min_samples = gr.Slider(
                    label="Min Samples per Token",
                    minimum=PROCESSING_SETTINGS["min_samples_range"][0],
                    maximum=PROCESSING_SETTINGS["min_samples_range"][1],
                    value=PROCESSING_SETTINGS["default_min_samples"],
                    step=5,
                    visible=False,
                    info="Minimum perplexity samples to collect per token (encoder only)",
                )

            analyze_btn = gr.Button(
                "🔍 Analyze Perplexity", variant="primary", size="lg"
            )

        with gr.Column(scale=3):
            summary_output = gr.Markdown(label="Summary")
            viz_output = gr.HTML(label="Perplexity Visualization")

    # Full-width table
    with gr.Row():
        table_output = gr.Dataframe(
            label="Detailed Token Results", interactive=False, wrap=True
        )

    # Update model dropdown based on type selection
    def update_model_choices(model_type):
        return gr.update(
            choices=DEFAULT_MODELS[model_type], value=DEFAULT_MODELS[model_type][0]
        )

    def toggle_advanced_settings(model_type):
        is_encoder = model_type == "encoder"
        return [
            gr.update(visible=is_encoder),  # mask_probability
            gr.update(visible=is_encoder),  # min_samples
        ]

    model_type.change(
        fn=lambda mt: [update_model_choices(mt)] + toggle_advanced_settings(mt),
        inputs=[model_type],
        outputs=[model_name, mask_probability, min_samples],
    )

    # Set up the analysis function
    analyze_btn.click(
        fn=process_text,
        inputs=[text_input, model_name, model_type, mask_probability, min_samples],
        outputs=[summary_output, viz_output, table_output],
    )

    # Add examples
    with gr.Accordion("📝 Example Texts", open=False):
        examples_data = [
            [
                ex["text"],
                ex["model"],
                ex["type"],
                ex.get("mask_prob", 0.15),
                ex.get("min_samples", 10),
            ]
            for ex in UI_SETTINGS["examples"]
        ]

        gr.Examples(
            examples=examples_data,
            inputs=[text_input, model_name, model_type, mask_probability, min_samples],
            outputs=[summary_output, viz_output, table_output],
            fn=process_text,
            cache_examples=False,
            label="Click on an example to try it out:",
        )

    # Add footer with information
    gr.Markdown("""
    ---

    ### 📊 How it works:

    - **Decoder Models** (GPT, etc.): Calculate true perplexity by measuring how well the model predicts the next token
    - **Encoder Models** (BERT, etc.): Calculate pseudo-perplexity using statistical sampling with multiple token masking
    - **Mask Probability**: For encoder models, controls what fraction of tokens get masked in each iteration
    - **Min Samples**: Minimum number of perplexity measurements collected per token for robust statistics
    - **Color Coding**: Red = High perplexity (uncertain), Green = Low perplexity (confident)

    ### ⚠️ Notes:
    - First model load may take some time
    - Models are cached after first use
    - Very long texts are truncated to 512 tokens
    - GPU acceleration is used when available
    - Encoder models use Monte Carlo sampling for robust perplexity estimates
    - Higher min samples = more accurate but slower analysis
    """)

if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
    except Exception as e:
        print(f"❌ Failed to launch app: {e}")
        print("💡 Try running with: python run.py")
        # Fallback to basic launch
        try:
            demo.launch()
        except Exception as fallback_error:
            print(f"❌ Fallback launch also failed: {fallback_error}")
            print("💡 Try updating Gradio: pip install --upgrade gradio")
