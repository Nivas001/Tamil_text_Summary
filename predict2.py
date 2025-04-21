import spacy
from pathlib import Path
import sys

# --- Configuration ---
# Ensure this path points to your best trained model directory
# We are using the one trained on the CPU from the previous steps.
MODEL_PATH = Path("./training_400/model-best")
# --- End Configuration ---

def load_model(path):
    """Loads the spaCy model."""
    if not path.exists():
        print(f"✘ Error: Model directory not found at {path.resolve()}")
        print("Please ensure the path is correct and you have trained the model.")
        sys.exit(1)
    try:
        # The CuPy warnings might still appear here if CUDA PATH isn't set,
        # but loading should proceed using CPU for this model.
        nlp = spacy.load(path)
        print(f"\n✔ Successfully loaded model from: {path.resolve()}")
        return nlp
    except Exception as e:
        print(f"✘ Error loading model from {path.resolve()}: {e}")
        print("Please ensure the model path is correct and the model files are intact (especially meta.json).")
        sys.exit(1) # Exit if model can't be loaded

def predict_entities(nlp, text):
    """Processes text and prints found entities."""
    if not text or text.isspace():
        print("Input text is empty.")
        return

    # Limit display length for very long inputs in the prompt message
    display_text = f"\"{text[:100]}...\"" if len(text) > 100 else f"\"{text}\""
    print(f"\n---> Processing text: {display_text}")

    # Process the text with the loaded NLP model
    doc = nlp(text)

    # Check if any entities were found
    if doc.ents:
        print("\n--- Entities Found ---")
        for ent in doc.ents:
            print(f"  Text:  '{ent.text}'")
            print(f"  Label: {ent.label_}")
            print(f"  Start: {ent.start_char}, End: {ent.end_char}")
            print("-" * 25) # Separator between entities
    else:
        print("\n--- No entities found in this text. ---")
    print("=" * 40) # Separator between different predictions

def main():
    """Main function to load model and run interactive prediction loop."""
    nlp_model = load_model(MODEL_PATH)

    print("\n==============================")
    print("  Interactive NER Predictor")
    print("==============================")
    print(f"Model loaded: {MODEL_PATH.name}")
    print("Enter Tamil text below to identify entities.")
    print("Type 'quit' or 'exit' (or just press Enter on an empty line) to stop.")
    print("-" * 40)

    while True:
        try:
            # Get input from the user
            user_input = input("Enter text >> ")

            # Check for exit conditions
            if user_input.lower() in ["quit", "exit", ""]:
                print("\nExiting predictor.")
                break

            # Perform prediction
            predict_entities(nlp_model, user_input)

        except EOFError: # Handle Ctrl+D if used in some terminals
            print("\nExiting predictor.")
            break
        except KeyboardInterrupt: # Handle Ctrl+C cleanly
            print("\nExiting predictor.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            # Optionally continue or break based on error severity
            # break


if __name__ == "__main__":
    main()