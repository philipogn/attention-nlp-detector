import pandas as pd
import numpy as np
import torch
import re # regular expression
import gc # garbage collector
from joblib import dump # model save to prevent retraining
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import preprocess_text, analyze_prompt

datasets = [
    "data/train/guychuk_data5000.csv",
    "data/train/harelix_dataset.csv",
    "data/train/GPT_generated_dataset.csv",
    "data/train/deepset_train.csv"
]

MODEL_NAME = "meta-llama/Llama-3.2-1B"
# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, output_attentions=True, return_dict_in_generate=True, attn_implementation="eager", device_map="auto"
)
# model.to("cpu")

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

def extract_features(data):
    entropy_scores = []
    variance_scores = []
    for prompt in data['prompt']:
        entropy_score, variance_score = analyze_prompt(prompt, tokenizer, model)
        entropy_scores.append(entropy_score)
        variance_scores.append(variance_score)
        label = data.loc[data['prompt'] == prompt, 'label'].values[0]
        print(f"{prompt[:30]}...,{label}, Entropy:{entropy_score:.4f}, Variance:{variance_score:.4f}")
        torch.cuda.empty_cache()
    data["entropy"] = entropy_scores
    data["variance"] = variance_scores
    return data

def random_forest_train(df):
    df['prompt'] = df['prompt'].apply(preprocess_text)
    
    # prepare training data
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(df['prompt'])
    X = np.hstack((X_tfidf.toarray(), df[['entropy', 'variance']].values))
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rand_forest = RandomForestClassifier()
    rand_forest.fit(X_train, y_train)

    # predictions and metrics
    y_pred = rand_forest.predict(X_test)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    # metrics results
    print(f"\nRandom Forest Model Evaluation on predicting test set on {MODEL_NAME}:")
    print(f"Precision: {precision} | Recall: {recall} | F1 Score: {f1}")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return rand_forest, tfidf

def log_reg_train(df):
    df['prompt'] = df['prompt'].apply(preprocess_text)

    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(df['prompt'])

    scaler = StandardScaler()
    df[['entropy', 'variance']] = scaler.fit_transform(df[['entropy', 'variance']])

    X = np.hstack((X_tfidf.toarray(), df[['entropy', 'variance']].values))
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    log_reg = LogisticRegression(class_weight='balanced')
    log_reg.fit(X_train, y_train)

    # predictions and metrics
    y_pred = log_reg.predict(X_test)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    # metrics results
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nLogistic Regression Model Evaluation on predicting test set with {MODEL_NAME}:")
    print(f"Precision: {precision} | Recall: {recall} | F1 Score: {f1}")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return log_reg, tfidf, scaler

if __name__ == "__main__":
    # load all datasets and process them
    load_df = load_datasets(datasets)
    df = extract_features(load_df)

    rand_forest, rand_tfidf = random_forest_train(df)
    log_reg, log_tfidf, scaler = log_reg_train(df)
    
    # save random forest model
    with open(f'{MODEL_NAME}_rand_model.pkl', 'wb') as file:
        dump((rand_forest, rand_tfidf), file, compress=3) # no scaler required

    # save log reg model
    with open(f'{MODEL_NAME}_log_model.pkl', 'wb') as file:
        dump((log_reg, log_tfidf, scaler), file, compress=3)  # Compress to reduce memory usage

