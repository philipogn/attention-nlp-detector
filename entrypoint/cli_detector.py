import os
import numpy as np
import pandas as pd
from joblib import load
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils import analyze_prompt

MODEL_OPTIONS = {
    "1": ("meta-llama/Llama-3.2-1B", "llama"),
    "2": ("Qwen/Qwen2.5-1.5B-Instruct", "qwen"),
    "3": ("microsoft/Phi-3-mini-128k-instruct", "phi")
}

def load_model(model_name):
    print(f"\nLoading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True, return_dict_in_generate=True, attn_implementation="eager", device_map="auto"
    )
    return tokenizer, model


def predict_prompt(prompt, model_alias, tokenizer, model):
    model_file = f"saved_models/{model_alias}_svm_model.pkl"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found.")
    else:
        with open(model_file, 'rb') as file:
            loaded_model, loaded_tfidf, loaded_scaler = load(file)
    entropy, variance = analyze_prompt(prompt, tokenizer, model)
    tfidf_features = loaded_tfidf.transform([prompt]).toarray()
    scaled = loaded_scaler.transform(pd.DataFrame([[entropy, variance]], columns=['entropy', 'variance']))
    features = np.hstack((tfidf_features, scaled))

    prediction = loaded_model.predict(features)
    return prediction, entropy, variance


def run_cli():

    print("\n=== Prompt Injection Detection CLI ===\n")
    print("Choose a base language model:")  
    for key, (name, _) in MODEL_OPTIONS.items():
        print(f"{key}: {name}")
    while True:
        model_choice = input("Enter model number: ").strip()
        if model_choice not in MODEL_OPTIONS:
            print("\nInvalid model selection, please select a number 1, 2 or 3.")
            continue
        else:
            break

    model_name, model_alias = MODEL_OPTIONS[model_choice]
    tokenizer, model = load_model(model_name)

    while True:
        prompt = input("\nEnter your prompt (Enter 'exit()' to quit):\n> ")
        if not prompt:
            print("Cannot be empty, please enter a prompt.")
            continue
        elif prompt == "exit()":
            print("Exiting CLI.")
            break
        else:
            print("\nAnalysing prompt...\n")
            try:
                prediction, entropy, variance = predict_prompt(prompt, model_alias, tokenizer, model)
                if prediction == 1:
                    label_str = "MALICIOUS PROMPT DETECTED"
                else:
                    label_str = "Benign prompt detected"
                print(f"Prediction: {label_str}")
                print(f"Entropy Score: {entropy:.4f}, Variance Score: {variance:.4f}")
            except Exception as e:
                print(f"Error during prediction: {e}")

if __name__ == "__main__":
    run_cli()
