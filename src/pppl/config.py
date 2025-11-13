# Configuration file for PerplexityViewer

# Default models for different types
DEFAULT_MODELS = {
    "decoder": [
        "gpt2",
        "distilgpt2",
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "openai-gpt",
    ],
    "encoder": [
        "bert-base-uncased",
        "bert-base-cased",
        "distilbert-base-uncased",
        "roberta-base",
        "UMCU/CardioMedRoBERTa.nl",
        "UMCU/CardioBERTa_base.nl",
        "UMCU/CardioBERTa.nl_clinical",
        "UMCU/CardioDeBERTa.nl",
        "UMCU/CardioDeBERTa.nl_clinical",
        "CLTL/MedRoBERTa.nl",
        "DTAI-KULeuven/robbert-2023-dutch-base",
        "DTAI-KULeuven/robbert-2023-dutch-large",
    ],
}

# Model display settings
MODEL_SETTINGS = {"max_length": 512, "torch_dtype": "float16", "device_map": "auto"}

# Visualization settings
VIZ_SETTINGS = {
    "max_perplexity_display": 50.0,  # Cap visualization at this perplexity value
    "color_scheme": {
        "low_perplexity": {
            "r": 46,
            "g": 204,
            "b": 113,
        },  # Green for low perplexity (confident)
        "medium_perplexity": {
            "r": 241,
            "g": 196,
            "b": 15,
        },  # Yellow for medium perplexity
        "high_perplexity": {
            "r": 231,
            "g": 76,
            "b": 60,
        },  # Red for high perplexity (uncertain)
        "background_alpha": 0.7,  # Background transparency
        "border_alpha": 0.9,  # Border transparency
    },
    "thresholds": {
        "low_threshold": 0.3,  # Below this is low perplexity (green)
        "high_threshold": 0.7,  # Above this is high perplexity (red)
    },
    "displacy_options": {"ents": ["PP"], "colors": {}},
}

# Processing settings
PROCESSING_SETTINGS = {
    "epsilon": 1e-10,  # Small value to avoid log(0)
    "default_mask_probability": 0.15,
    "min_mask_probability": 0.05,
    "max_mask_probability": 0.5,
    "default_min_samples": 10,
    "min_samples_range": (5, 50),
}

# UI settings
UI_SETTINGS = {
    "theme": "soft",
    "title": "📈 Perplexity Viewer",
    "description": """
    Visualize per-token perplexity using color gradients.
    - **Red**: High perplexity (model is uncertain)
    - **Green**: Low perplexity (model is confident)

    Choose between decoder models (like GPT) for true perplexity or encoder models (like BERT) for pseudo-perplexity via MLM.
    """,
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

# Error messages
ERROR_MESSAGES = {
    "empty_text": "Please enter some text to analyze.",
    "model_load_error": "Error loading model {model_name}: {error}",
    "processing_error": "Error processing text: {error}",
    "no_tokens_masked": "No tokens were masked during MLM processing.",
    "invalid_model_type": "Invalid model type. Must be 'encoder' or 'decoder'.",
}
