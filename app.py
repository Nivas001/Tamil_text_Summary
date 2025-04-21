# -*- coding: utf-8 -*-

# --- Prerequisites ---
# Ensure these are in your requirements.txt for Hugging Face Spaces:
# spacy==3.5.0 # Or the version used to train NER model
# transformers>=4.20.0
# torch>=1.10.0 # Or tensorflow
# sentencepiece>=0.1.90
# protobuf==3.20.3
# datasets # Often needed by transformers/evaluate
# evaluate # If using compute_metrics (not strictly needed for this app)
# gradio>=3.0.0
# numpy
# accelerate # Good practice for transformers

import spacy
from pathlib import Path
import sys
import gradio as gr # Import Gradio
import warnings
import re
import numpy as np
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
except ImportError:
    print("✘ Error: 'transformers' or 'torch' library not found.")
    print("Ensure they are listed in requirements.txt")
    # Gradio might handle showing an error in the UI, but good to log.
    # We'll handle model loading failure later.
    pass


# --- Configuration ---
# 1. Path to your spaCy NER model directory WITHIN THE SPACE REPO
#    (Upload your model-best folder and adjust path if needed)
NER_MODEL_PATH = Path("./model-best") # Assumes model-best is at the repo root

# 2. Hugging Face model name for mT5 summarization
SUMMARIZATION_MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

# 3. Device Selection (CPU is default/safer for free HF Spaces)
DEVICE = "cpu"
# Uncomment below if using GPU hardware on Spaces and CUDA is confirmed working there
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4. Summarization parameters
SUMM_NUM_BEAMS = 4
MIN_LEN_PERC = 0.30 # Target minimum summary length as % of input tokens
MAX_LEN_PERC = 0.75 # Target maximum summary length as % of input tokens
ABS_MIN_TOKEN_LEN = 30 # Absolute minimum token length
ABS_MAX_TOKEN_LEN = 512 # Absolute maximum token length (Adjust based on model/needs)
# --- End Configuration ---

warnings.filterwarnings("ignore", message="CUDA path could not be detected*")
warnings.filterwarnings("ignore", message=".*You are using `torch.load` with `weights_only=False`.*")
warnings.filterwarnings("ignore", message=".*The sentencepiece tokenizer that you are converting.*")

# --- Global Variables for Loaded Models (Load Once) ---
ner_model_global = None
summ_tokenizer_global = None
summ_model_global = None
models_loaded = False

# --- Model Loading Functions (Modified slightly for global loading) ---
def load_ner_model(path):
    """Loads the spaCy NER model and ensures sentencizer is present."""
    global ner_model_global # Declare intent to modify global variable
    if not path.exists():
        print(f"✘ FATAL: NER Model directory not found at {path.resolve()}")
        return False
    try:
        ner_model_global = spacy.load(path)
        print(f"✔ Successfully loaded NER model from: {path.resolve()}")
        # Ensure a sentence boundary detector is present
        component_to_add_before = None
        if "tok2vec" in ner_model_global.pipe_names: component_to_add_before="tok2vec"
        elif "ner" in ner_model_global.pipe_names: component_to_add_before="ner"
        if not ner_model_global.has_pipe("sentencizer") and not ner_model_global.has_pipe("parser"):
            try:
                if component_to_add_before: ner_model_global.add_pipe("sentencizer", before=component_to_add_before)
                else: ner_model_global.add_pipe("sentencizer", first=True)
                print("INFO: Added 'sentencizer' to loaded NER pipeline.")
            except Exception as e_pipe:
                print(f"✘ WARNING: Could not add 'sentencizer': {e_pipe}. Sentence splitting might fail.")
        return True
    except Exception as e:
        print(f"✘ FATAL: Error loading NER model from {path.resolve()}: {e}")
        return False

def load_summarizer(model_name):
    """Loads the Hugging Face tokenizer and model for summarization."""
    global summ_tokenizer_global, summ_model_global # Declare intent to modify globals
    try:
        print(f"\nLoading summarization tokenizer: {model_name}...")
        summ_tokenizer_global = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading summarization model: {model_name}...")
        summ_model_global = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summ_model_global.to(DEVICE)
        # Optional: Override max length config (can be unstable, test carefully)
        # try:
        #     summ_model_global.config.max_length = ABS_MAX_TOKEN_LEN
        #     print(f"INFO: Overrode model config max_length to {ABS_MAX_TOKEN_LEN}")
        # except Exception as e_cfg:
        #     print(f"WARN: Could not override model config max_length: {e_cfg}")
        print(f"INFO: Model's default configured max generation length: {summ_model_global.config.max_length}")
        print(f"✔ Successfully loaded summarization model '{model_name}' on {DEVICE}.")
        return True
    except Exception as e:
        print(f"✘ FATAL: Error loading summarization model '{model_name}': {e}")
        return False

# --- Load models when the script starts ---
print("Application starting up... Loading models...")
models_loaded = load_ner_model(NER_MODEL_PATH) and load_summarizer(SUMMARIZATION_MODEL_NAME)
if models_loaded:
     print("\n--- All models loaded successfully! Ready for input. ---")
else:
     print("\n✘✘✘ CRITICAL ERROR: Model loading failed. The application might not work correctly. Check logs. ✘✘✘")


# --- Core Logic Functions (Keep as they were) ---
def summarize_text(tokenizer, model, text, num_beams=SUMM_NUM_BEAMS,
                   min_length_perc=MIN_LEN_PERC, max_length_perc=MAX_LEN_PERC):
    """Generates abstractive summary with length based on input token percentage."""
    if not text or text.isspace(): return "Input text is empty."
    print("INFO: Generating summary (using percentage lengths)...") # Use print for logs
    try:
        # 1. Calculate input token length
        input_ids = tokenizer(text, return_tensors="pt", truncation=False, padding=False).input_ids
        input_token_count = input_ids.shape[1]
        if input_token_count == 0: return "Input text tokenized to zero tokens."
        print(f"INFO: Input has {input_token_count} tokens.")

        # 2. Calculate target token lengths
        min_len_tokens = int(input_token_count * min_length_perc)
        max_len_tokens = int(input_token_count * max_length_perc)

        # 3. Apply absolute limits and ensure min < max
        min_len_tokens = max(ABS_MIN_TOKEN_LEN, min_len_tokens)
        max_len_tokens = max(min_len_tokens + 10, max_len_tokens)
        max_len_tokens = min(ABS_MAX_TOKEN_LEN, max_len_tokens)
        min_len_tokens = min(min_len_tokens, max_len_tokens)
        print(f"INFO: Target summary token length: min={min_len_tokens}, max={max_len_tokens}.")

        # 4. Tokenize for model input
        inputs = tokenizer(text, max_length=1024, return_tensors="pt", padding="max_length", truncation=True).to(DEVICE)

        # 5. Generate summary
        summary_ids = model.generate(inputs['input_ids'],
                                     num_beams=num_beams,
                                     max_length=max_len_tokens,
                                     min_length=min_len_tokens,
                                     early_stopping=True)

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print("✔ Summary generation complete.")
        return summary
    except Exception as e:
        print(f"✘ Error during summary generation: {e}")
        return "[Error during summary generation]"

def extract_entities(ner_nlp, text):
    """Extracts named entities using the spaCy NER model."""
    if not text or text.isspace(): return []
    print("INFO: Extracting entities...")
    try:
        doc = ner_nlp(text)
        entities = list({(ent.text.strip(), ent.label_) for ent in doc.ents if ent.text.strip()})
        print(f"✔ Extracted {len(entities)} unique entities.")
        return entities
    except Exception as e:
        print(f"✘ Error during entity extraction: {e}")
        return []

def create_prompted_input(text, entities):
    """Creates a new input string with unique entities prepended."""
    if not entities:
        return text
    unique_entity_texts = sorted(list({ent[0] for ent in entities if ent[0]}))
    entity_string = ", ".join(unique_entity_texts)
    separator = ". முக்கிய சொற்கள்: "
    prompted_text = f"{entity_string}{separator}{text}"
    print(f"INFO: Created prompted input with {len(unique_entity_texts)} unique entities.")
    return prompted_text

# --- Main Processing Function for Gradio ---
def process_text_for_gradio(input_paragraph):
    """Takes input text and returns standard summary and NER-enhanced output string."""
    # Check if models loaded correctly
    if not models_loaded or ner_model_global is None or summ_tokenizer_global is None or summ_model_global is None:
        error_msg = "[FATAL ERROR: Models did not load correctly. Check application logs.]"
        return error_msg, error_msg

    text_to_process = input_paragraph.strip()
    if not text_to_process:
        return "(No input text provided)", "(No input text provided)"

    # --- Generate Output 1: Standard Summary ---
    standard_summary = summarize_text(
        summ_tokenizer_global, summ_model_global, text_to_process,
        num_beams=SUMM_NUM_BEAMS
    )

    # --- Prepare Output 2: NER Analysis + NER-Influenced Summary ---
    # a) Extract entities
    extracted_entities = extract_entities(ner_model_global, text_to_process)

    # b) Create prompted input
    prompted_input_text = create_prompted_input(text_to_process, extracted_entities)

    # c) Generate summary from prompted input
    ner_influenced_summary = summarize_text(
        summ_tokenizer_global, summ_model_global, prompted_input_text,
        num_beams=SUMM_NUM_BEAMS
    )

    # d) Format the combined Output 2 string
    output2_lines = ["--- Key Entities Found by NER ---"]
    if extracted_entities:
        for text_ent, label in extracted_entities:
            output2_lines.append(f"- '{text_ent}' ({label})")
    else:
        output2_lines.append("(No entities found by NER model)")

    output2_lines.append("\n--- NER-Influenced Summary ---")
    output2_lines.append(ner_influenced_summary)
    output2_lines.append("\n(NOTE: Compare with Output 1. Prepending entities is experimental.)")

    output2_display = "\n".join(output2_lines)

    # Return the two outputs for Gradio
    return standard_summary, output2_display


# --- Create and Launch Gradio Interface ---
print("\nSetting up Gradio interface...")
# Add description specific to your setup
app_description = """
ஒரு தமிழ் பத்தியை உள்ளிடவும். இந்த பயன்பாடு இரண்டு சுருக்கங்களை உருவாக்கும்:
1.  **நிலையான சுருக்கம்:** முன் பயிற்சி பெற்ற mT5 மாதிரியைப் பயன்படுத்தி உருவாக்கப்பட்டது.
2.  **NER பகுப்பாய்வு & செல்வாக்கு பெற்ற சுருக்கம்:** உங்கள் தனிப்பயன் NER மாதிரியால் அடையாளம் காணப்பட்ட முக்கிய சொற்களைப் பட்டியலிடுகிறது, பின்னர் அந்த சொற்களை உள்ளீட்டின் முன்சேர்த்து உருவாக்கப்பட்ட சுருக்கத்தைக் காட்டுகிறது (இது சுருக்கத்தில் அவற்றைச் சேர்க்க மாதிரியை பாதிக்கலாம்).

Enter a Tamil paragraph. This app generates two summaries:
1.  **Standard Summary:** Generated using the pre-trained mT5 model.
2.  **NER Analysis & Influenced Summary:** Lists key entities identified by your custom NER model, then shows a summary generated by prepending those entities to the input (which may influence the model to include them).
"""

# Add examples if desired
example_list = [
    ["இந்திய கிரிக்கெட் அணியின் முன்னாள் கேப்டனும், சென்னை சூப்பர் கிங்ஸ் அணியின் தற்போதைய கேப்டனுமான எம்.எஸ். தோனி ஐபிஎல் தொடரில் இருந்து ஓய்வு பெறுவதாக வெளியான தகவல்கள் வெறும் வதந்தி என சிஎஸ்கே நிர்வாகம் மறுத்துள்ளது. நேற்று முன்தினம் மும்பை இந்தியன்ஸ் அணிக்கு எதிரான போட்டியில் சென்னை அணி அபார வெற்றி பெற்றது. இதில் தோனியின் கடைசி நேர அதிரடி ஆட்டம் முக்கிய பங்கு வகித்தது."],
    ["ஜெய்ப்பூர்: ஐபிஎல் 2025 ஆம் ஆண்டு சீசனில் ராஜஸ்தான் ராயல்ஸ் அணிக்காக 14 வயது சூரியவன்ஷி அறிமுகமானார். இதன் மூலம் இளம் வயதில் ஐபிஎல் தொடரில் களமிறங்கிய வீரர் என்ற சாதனையை வைபவ் படைத்திருக்கிறார்."]
]


iface = gr.Interface(
    fn=process_text_for_gradio, # The function to call
    inputs=gr.Textbox(lines=15, label=" உள்ளீடு தமிழ் பத்தி (Input Tamil Paragraph)"),
    outputs=[
        gr.Textbox(label=" வெளியீடு 1: நிலையான சுருக்கம் (Output 1: Standard Summary)"),
        gr.Textbox(label=" வெளியீடு 2: NER பகுப்பாய்வு & செல்வாக்கு பெற்ற சுருக்கம் (Output 2: NER Analysis & Influenced Summary)")
    ],
    title="தமிழ் சுருக்கம் மற்றும் NER ஒருங்கிணைப்பு (Tamil Summarization + NER Integration)",
    description=app_description,
    allow_flagging='never',
    examples=example_list
)

print("Launching Gradio interface... Access it at the URL provided.")
# queue() enables handling multiple simultaneous users
# share=True creates a temporary public link (use False for local only)
iface.launch(show_error=True)