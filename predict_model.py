import pandas as pd
import numpy as np
from joblib import load
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import analyze_prompt

DATASET_TEST = [
    # "data/test/deepset_test.csv",
    "data/test/promptshield_dataset.csv",
    # "data/test/jackhhao_jailbreak_dataset.csv",
    # "data/test/wild_awesome_gen_test.csv"
]

MODEL_NAME = "meta-llama/Llama-3.2-1B"
# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"

# MAYBE PUT THIS INTO A FUNCTION???
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, output_attentions=True, return_dict_in_generate=True, attn_implementation="eager", device_map="auto"
)

def load_datasets(dataset_paths):
    dataframes = []
    # load datasets and concatenate into a single dataframe
    for path in dataset_paths:
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
                dataframes.append(df)
            else:
                print(f"Skipping unsupported file: {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    if not dataframes:
        raise ValueError("No valid datasets found!")
    
    return pd.concat(dataframes, ignore_index=True)

def predict_prompt_logreg(prompts, saved_model_name):
    predictions = []
    with open(f'saved_models/{saved_model_name}_log_model.pkl', 'rb') as file:
        loaded_model, loaded_tfidf, loaded_scaler = load(file)

    for prompt in prompts:
        entropy_score, variance_score = analyze_prompt(prompt, tokenizer, model)
        prompt_tfidf = loaded_tfidf.transform([prompt]).toarray()
        scaled_features = loaded_scaler.transform(pd.DataFrame([[entropy_score, variance_score]], columns=['entropy', 'variance'])) # for logreg
        features = np.hstack((prompt_tfidf, scaled_features))
        prediction = loaded_model.predict(features)
        predictions.append(prediction[0])
    return predictions

def predict_prompt_svm(prompts, saved_model_name):
    predictions = []
    with open(f'saved_models/{saved_model_name}_svm_model.pkl', 'rb') as file:
        loaded_model, loaded_tfidf, loaded_scaler = load(file)

    for prompt in prompts:
        entropy_score, variance_score = analyze_prompt(prompt, tokenizer, model)
        prompt_tfidf = loaded_tfidf.transform([prompt]).toarray()
        scaled_features = loaded_scaler.transform(pd.DataFrame([[entropy_score, variance_score]], columns=['entropy', 'variance'])) # for svm
        features = np.hstack((prompt_tfidf, scaled_features))
        prediction = loaded_model.predict(features)
        predictions.append(prediction[0])
    return predictions

def predict_prompt_randforest(prompts, saved_model_name):
    predictions = []
    with open(f'saved_models/{saved_model_name}_rand_model.pkl', 'rb') as file:
        loaded_model, loaded_tfidf = load(file)

    for prompt in prompts:
        entropy_score, variance_score = analyze_prompt(prompt, tokenizer, model)
        prompt_tfidf = loaded_tfidf.transform([prompt]).toarray()
        features = np.hstack((prompt_tfidf, np.array([[entropy_score, variance_score]]))) # no scaling needed for random forest
        prediction = loaded_model.predict(features)
        predictions.append(prediction[0])
    return predictions

def evaluation_metrics(true_labels, predicted_labels, model_type):
    # calculate metrics and print
    precision_unseen = metrics.precision_score(true_labels, predicted_labels)
    recall_unseen = metrics.recall_score(true_labels, predicted_labels)
    f1_unseen = metrics.f1_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print(f"\nPredicting on unseen data with {MODEL_NAME} ({model_type})")
    print(f"Precision: {precision_unseen} | Recall: {recall_unseen} | F1: {f1_unseen}")
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report: ", classification_report(true_labels, predicted_labels))


if __name__ == "__main__":
    df = load_datasets(DATASET_TEST)
    prompts = df["prompt"].tolist()
    true_labels = df["label"].astype(int).tolist()

    models = {
        "llama": ("meta-llama/Llama-3.2-1B",),
        "qwen": ("Qwen/Qwen2.5-1.5B-Instruct",),
        "phi": ("microsoft/Phi-3-mini-128k-instruct")
    }
    
    saved_model_name = "llama"  # change to load saved language models: llama, qwen, phi

    # logistic regression
    predicted_labels = predict_prompt_logreg(prompts, saved_model_name)
    evaluation_metrics(true_labels, predicted_labels, "LOGISTIC REGRESSION")

    # svm
    predicted_labels_svm = predict_prompt_svm(prompts, saved_model_name)
    evaluation_metrics(true_labels, predicted_labels_svm, "SVM")

    # random forest
    predicted_labels_rand = predict_prompt_randforest(prompts, saved_model_name)
    evaluation_metrics(true_labels, predicted_labels_rand, "RANDOM FOREST")
