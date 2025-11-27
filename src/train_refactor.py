import pandas as pd
import numpy as np
import torch
from joblib import dump # model save to prevent retraining
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import yaml

class TrainConfig():
    MODEL_OPTIONS = {
        "llama": "meta-llama/Llama-3.2-1B",
        "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
        "phi": "microsoft/Phi-3-mini-128k-instruct"
    }

    DATASET_TRAIN = [
        "data/train/guychuk_data5000.csv",
        "data/train/harelix_dataset.csv",
        "data/train/GPT_generated_dataset.csv",
        "data/train/deepset_train.csv"
    ]

class TrainModel(TrainConfig):
    def __init__(self, model_name):
        self.model_name = model_name
        self.tfidf = TfidfVectorizer(lowercase=True)
        self.scaler = StandardScaler()

    def support_vector_machine(self, df):
        X_tfidf = self.tfidf.fit_transform(df['prompt'])
        df[['entropy', 'variance']] = self.scaler.fit_transform(df[['entropy', 'variance']])

        X = np.hstack((X_tfidf.toarray(), df[['entropy', 'variance']].values))
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        svm_cls = svm.SVC(kernel="linear")
        svm_cls.fit(X_train, y_train)

        y_pred = svm_cls.predict(X_test)
        self.evaluate(y_test, y_pred)
        return svm_cls, self.tfidf, self.scaler
    
    def evaluate(self, y_test, y_pred):
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nSupport Vector Machine metrics on predicting test set with {self.model_name}:")
        print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | Accuracy: {accuracy:.2f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
    
    def run(self):
        svm_cls, svm_tfidf, svm_scaler = self.support_vector_machine(data_extracted)
        self.save_model(svm_cls, svm_tfidf, svm_scaler)
    
    def save_model(self, svm_cls, svm_tfidf, svm_scaler):
        with open('svm_model.pkl', 'wb') as file:
            dump((svm_cls, svm_tfidf, svm_scaler), file, compress=3)


if __name__ == "__main__":
    config = yaml.safe_load(open('config/config.yaml'))
    train = TrainModel('llama')

    train

