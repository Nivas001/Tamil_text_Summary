# -*- coding: utf-8 -*-

import spacy
from pathlib import Path
import sys
# Make sure you have installed transformers, torch, sentencepiece, spacy, protobuf==3.20.3
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    print("✘ Error: 'transformers' library not found.")
    print("Please install it: pip install transformers torch sentencepiece")
    sys.exit(1)
import torch
import warnings
import re # For slightly better entity checking
import numpy as np # Needed for calculation

# --- Configuration ---
# 1. Path to your trained spaCy NER model (Use your best one!)
NER_MODEL_PATH = Path("./training_400/model-best") # <-- ADJUST TO YOUR BEST NER MODEL

# 2. Hugging Face model name for mT5 summarization
SUMMARIZATION_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

# 3. Device: "cuda" for GPU or "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4. Summarization parameters
SUMM_NUM_BEAMS = 4
# --- NEW: Percentage-based length ---
MIN_LEN_PERC = 0.30 # Target minimum summary length as % of input tokens (e.g., 30%)
MAX_LEN_PERC = 0.75 # Target maximum summary length as % of input tokens (e.g., 55%)
# --- NEW: Absolute token limits (safety net) ---
ABS_MIN_TOKEN_LEN = 20 # Don't generate summaries shorter than this many tokens
ABS_MAX_TOKEN_LEN = 512 # Don't generate summaries longer than this many tokens
# --- End Configuration ---

warnings.filterwarnings("ignore", message="CUDA path could not be detected*")
warnings.filterwarnings("ignore", message=".*You are using `torch.load` with `weights_only=False`.*")

# --- Model Loading Functions ---
# (Keep load_ner_model and load_summarizer functions exactly as in the previous corrected version)
def load_ner_model(path):
    """Loads the spaCy NER model and ensures sentencizer is present."""
    if not path.exists():
        print(f"✘ Error: NER Model directory not found at {path.resolve()}")
        sys.exit(1)
    try:
        nlp = spacy.load(path)
        print(f"✔ Successfully loaded NER model from: {path.resolve()}")
        # Ensure a sentence boundary detector is present
        component_to_add_before = None
        if "tok2vec" in nlp.pipe_names: component_to_add_before="tok2vec"
        elif "ner" in nlp.pipe_names: component_to_add_before="ner"
        if not nlp.has_pipe("sentencizer") and not nlp.has_pipe("parser"):
            try:
                if component_to_add_before: nlp.add_pipe("sentencizer", before=component_to_add_before)
                else: nlp.add_pipe("sentencizer", first=True)
                print("INFO: Added 'sentencizer' to loaded NER pipeline.")
            except Exception as e_pipe:
                print(f"✘ WARNING: Could not add 'sentencizer': {e_pipe}. Sentence splitting might fail.")
        return nlp
    except Exception as e:
        print(f"✘ Error loading NER model from {path.resolve()}: {e}")
        sys.exit(1)

def load_summarizer(model_name):
    """Loads the Hugging Face tokenizer and model for summarization."""
    try:
        print(f"\nLoading summarization tokenizer: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading summarization model: {model_name} (this may take time)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(DEVICE)
        try:
            new_max = 256  # Set your desired max length
            model.config.max_length = new_max
            print(f"INFO: Attempted to override model config max_length to {new_max}")
        except Exception as e_cfg:
            print(f"WARN: Could not override model config max_length: {e_cfg}")
        # return tokenizer, model
        print(f"INFO: Model's configured max generation length: {model.config.max_length}")
        print(f"✔ Successfully loaded summarization model '{model_name}' on {DEVICE}.")
        return tokenizer, model
    except Exception as e:
        print(f"✘ Error loading summarization model '{model_name}': {e}")
        print("Please ensure model name is correct, protobuf==3.20.3, internet access.")
        sys.exit(1)

# --- Core Logic Functions ---

# --- MODIFIED summarize_text function ---
def summarize_text(tokenizer, model, text, num_beams=SUMM_NUM_BEAMS,
                   min_length_perc=MIN_LEN_PERC, max_length_perc=MAX_LEN_PERC):
    """Generates abstractive summary with length based on input token percentage."""
    if not text or text.isspace(): return "Input text is empty."
    print("\nGenerating summary (using percentage lengths)...")
    try:
        # 1. Calculate input token length (important to NOT pad/truncate here)
        input_ids = tokenizer(text, return_tensors="pt", truncation=False, padding=False).input_ids
        input_token_count = input_ids.shape[1]
        if input_token_count == 0: return "Input text tokenized to zero tokens."
        print(f"INFO: Input text has approx {len(text.split())} words and {input_token_count} tokens.")

        # 2. Calculate target token lengths based on percentages
        min_len_tokens = int(input_token_count * min_length_perc)
        max_len_tokens = int(input_token_count * max_length_perc)

        # 3. Apply absolute limits and ensure min < max
        min_len_tokens = max(ABS_MIN_TOKEN_LEN, min_len_tokens) # Apply absolute minimum
         # Ensure max is reasonably larger than min, prevent max < min
        max_len_tokens = max(min_len_tokens + 10, max_len_tokens)
        # Apply absolute maximum (e.g., model limit or desired cap)
        max_len_tokens = min(ABS_MAX_TOKEN_LEN, max_len_tokens)
        # Ensure min_len is not greater than max_len after caps
        min_len_tokens = min(min_len_tokens, max_len_tokens)


        print(f"INFO: Target summary token length: min={min_len_tokens}, max={max_len_tokens}.")

        # 4. Tokenize *again* for model input (this time with padding/truncation to model max input size)
        # Max length here refers to the *input* sequence length limit for the model
        inputs = tokenizer(text, max_length=1024, return_tensors="pt", padding="max_length", truncation=True).to(DEVICE)

        # 5. Generate summary using CALCULATED min/max token lengths
        summary_ids = model.generate(inputs['input_ids'],
                                     num_beams=num_beams,
                                     max_length=max_len_tokens, # Use calculated max
                                     min_length=min_len_tokens, # Use calculated min
                                     early_stopping=True)

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print("✔ Summary generation complete.")
        return summary
    except Exception as e:
        print(f"✘ Error during summary generation: {e}")
        import traceback
        traceback.print_exc()
        return "[Error generating summary]"

# (Keep extract_entities function exactly as before)
def extract_entities(ner_nlp, text):
    """Extracts named entities using the spaCy NER model."""
    if not text or text.isspace(): return []
    print("\nExtracting entities from original text using custom NER model...")
    try:
        doc = ner_nlp(text)
        entities = list({(ent.text.strip(), ent.label_) for ent in doc.ents if ent.text.strip()}) # Unique entities
        print(f"✔ Extracted {len(entities)} unique entities.")
        return entities
    except Exception as e:
        print(f"✘ Error during entity extraction: {e}")
        return []

# (Keep create_prompted_input function exactly as before)
def create_prompted_input(text, entities):
    """Creates a new input string with entities prepended."""
    if not entities:
        print("INFO: No entities found by NER, using original text for prompted summary.")
        return text # Return original text if no entities found
    entity_string = ", ".join(ent[0] for ent in entities)
    separator = ". முக்கிய சொற்கள்: " # ". Key terms: "
    prompted_text = f"{entity_string}{separator}{text}"
    print(f"\nINFO: Created prompted input (showing start): {prompted_text[:250]}...") # For debugging
    return prompted_text

# --- Main execution ---
# (Keep main function exactly as before - it now calls the modified summarize_text)
def main():
    # Load models
    print("Loading models, please wait...")
    ner_model = load_ner_model(NER_MODEL_PATH)
    summ_tokenizer, summ_model = load_summarizer(SUMMARIZATION_MODEL_NAME)
    print("\nModels loaded successfully!")
    print("="*50)

    # Get Input Text from User
    print("Please paste the Tamil text paragraph you want to summarize below.")
    print("Press Enter after pasting the text.")
    print("(You might need to configure your terminal for multi-line paste if it's long)")
    print("-" * 50)
    input_paragraph = input("Input Text:\n") # Get input from user

    if not input_paragraph or input_paragraph.isspace():
        print("\n✘ Error: No input text provided. Exiting.")
        sys.exit(1)
    text_to_process = input_paragraph.strip()

    print("\n" + "="*50)
    print("Processing Input Text (Snippet):")
    print(text_to_process[:300] + "...")
    print("="*50)

    # --- Generate Output 1: Standard Summary (using percentage lengths) ---
    print("\n--- Output 1: Standard Abstractive Summary (Percentage Length) ---")
    standard_summary = summarize_text(
        summ_tokenizer, summ_model, text_to_process,
        num_beams=SUMM_NUM_BEAMS
        # Uses default percentages MIN_LEN_PERC, MAX_LEN_PERC from config section
    )
    print("\nStandard Summary:")
    print(standard_summary)
    print("-" * 50)

    # --- Generate Output 2: NER-Influenced Summary (using percentage lengths) ---
    print("\n--- Output 2: NER-Influenced Abstractive Summary (Percentage Length) ---")
    # a) Extract entities
    extracted_entities = extract_entities(ner_model, text_to_process)
    print("\nKey Entities Extracted by NER:")
    if extracted_entities:
        for text_ent, label in extracted_entities:
            print(f"  - '{text_ent}' ({label})")
    else:
        print("  No entities found by NER model.")

    # b) Create prompted input
    prompted_input_text = create_prompted_input(text_to_process, extracted_entities)

    # c) Generate summary from prompted input (using percentage lengths)
    ner_influenced_summary = summarize_text(
        summ_tokenizer, summ_model, prompted_input_text,
        num_beams=SUMM_NUM_BEAMS
         # Uses default percentages MIN_LEN_PERC, MAX_LEN_PERC from config section
    )
    print("\nNER-Influenced Summary (Generated using entities as prefix):")
    print(ner_influenced_summary)
    print("\nNOTE: Compare this summary with the standard summary (Output 1).")
    print("See if prepending entities influenced the output and included more of them.")
    print("This method is experimental and doesn't guarantee inclusion.")
    print("="*50)


if __name__ == "__main__":
    main()