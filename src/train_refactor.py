import pandas as pd
import numpy as np
import torch
from joblib import dump # model save to prevent retraining
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import analyze_prompt

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
        self.tokenizer = None
        self.model = None
        self.train_df = []
        self.tfidf = TfidfVectorizer(lowercase=True)
        self.scaler = StandardScaler()

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_OPTIONS[self.model_name])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_OPTIONS[self.model_name], output_attentions=True, return_dict_in_generate=True, attn_implementation="eager", device_map="auto"
        )
        return self.tokenizer, self.model

    def load_datasets(self):
        '''
        Load training datasets and concatenate into a single dataframe
        '''
        dataframes = []
        for path in self.DATASET_TRAIN:
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
        else:
            print(f"Loaded {len(dataframes)} datasets.")
        
        self.train_df = pd.concat(dataframes, ignore_index=True)
        return self.train_df

    def extract_features(self, data):
        entropy_scores = []
        variance_scores = []
        for prompt in self.train_df['prompt']:
            entropy_score, variance_score = analyze_prompt(prompt, self.tokenizer, self.model)
            entropy_scores.append(entropy_score)
            variance_scores.append(variance_score)
            label = self.train_df.loc[self.train_df['prompt'] == prompt, 'label'].values[0]
            print(f"{prompt[:30]}... | Label: {label} | Entropy: {entropy_score:.4f} | Variance: {variance_score:.4f}")
        self.train_df["entropy"] = entropy_scores
        self.train_df["variance"] = variance_scores
        return data

    def random_forest_train(self, df, model_name):    
        # prepare training data
        X_tfidf = self.tfidf.fit_transform(df['prompt'])
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
        print(f"\nRandom Forest metrics on predicting test set on {model_name}:")
        print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | Accuracy: {accuracy:.2f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        return rand_forest, self.tfidf

    def log_reg_train(self, df, model_name):
        X_tfidf = self.tfidf.fit_transform(df['prompt'])
        df[['entropy', 'variance']] = self.scaler.fit_transform(df[['entropy', 'variance']])

        X = np.hstack((X_tfidf.toarray(), df[['entropy', 'variance']].values))
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)

        # predictions and metrics
        y_pred = log_reg.predict(X_test)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nLogistic Regression metrics on predicting test set with {model_name}:")
        print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | Accuracy: {accuracy:.2f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        return log_reg, self.tfidf, self.scaler

    def support_vector_machine(self, df, model_name):
        X_tfidf = self.tfidf.fit_transform(df['prompt'])
        df[['entropy', 'variance']] = self.scaler.fit_transform(df[['entropy', 'variance']])

        X = np.hstack((X_tfidf.toarray(), df[['entropy', 'variance']].values))
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        svm_cls = svm.SVC(kernel="linear")
        svm_cls.fit(X_train, y_train)

        # predictions and metrics
        y_pred = svm_cls.predict(X_test)
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nSupport Vector Machine metrics on predicting test set with {model_name}:")
        print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | Accuracy: {accuracy:.2f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        return svm_cls, self.tfidf, self.scaler
    
    def run(self):
        model_name = MODEL_OPTIONS[selected_model]
        tokenizer, model = load_model_and_tokenizer(model_name)

        # load all datasets and process them
        load_df = load_datasets(DATASET_TRAIN)
        data_extracted = extract_features(load_df, tokenizer, model)

        rand_forest, rand_tfidf = random_forest_train(data_extracted, model_name)
        log_reg, log_tfidf, log_scaler = log_reg_train(data_extracted, model_name)
        svm_cls, svm_tfidf, svm_scaler = support_vector_machine(data_extracted, model_name)
    
    def save_models(self):
        # save random forest model
        with open('randforest_model.pkl', 'wb') as file:
            dump((rand_forest, rand_tfidf), file, compress=3) # no scaler required

        # save log reg model
        with open('logreg_model.pkl', 'wb') as file:
            dump((log_reg, log_tfidf, log_scaler), file, compress=3)  # Compress to reduce memory usage

        # save svm model
        with open('svm_model.pkl', 'wb') as file:
            dump((svm_cls, svm_tfidf, svm_scaler), file, compress=3)


if __name__ == "__main__":
    train = TrainModel('llama')

    selected_model = "llama"  # change to load saved language models: llama, qwen, phi


    


