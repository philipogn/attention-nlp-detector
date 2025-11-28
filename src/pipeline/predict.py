import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import yaml

class Predictor():
    def __init__(self, df):
        self.df = df
        self.prompts = None
        self.true_labels = None
        self.prediction = None

    def prepare_data(self):
        self.prompts = self.df['prompt'].tolist()
        self.true_labels = self.df['label'].astype(int).tolist()

    def predict_prompt_svm(self):
        with open(f'saved_models/llama_svm_model.pkl', 'rb') as file:
            loaded_model, loaded_tfidf, loaded_scaler = load(file)

        prompt_tfidf = loaded_tfidf.transform(self.df['prompt']).toarray()
        scaled_features = loaded_scaler.transform(self.df[['entropy', 'variance']])
        features = np.hstack((prompt_tfidf, scaled_features))
        self.prediction = loaded_model.predict(features)
        return self.prediction

    def evaluation_metrics(self):
        # calculate metrics and print
        precision_unseen = precision_score(self.true_labels, self.prediction)
        recall_unseen = recall_score(self.true_labels, self.prediction)
        f1_unseen = f1_score(self.true_labels, self.prediction)
        accuracy = accuracy_score(self.true_labels, self.prediction)
        
        # print(f"\nPredicting on unseen data with {model_name} ({classifier_type})")
        print(f"Precision: {precision_unseen:.2f} | Recall: {recall_unseen:.2f} | F1: {f1_unseen:.2f} | Accuracy: {accuracy:.2f}")
        print("Classification Report:\n", classification_report(self.true_labels, self.prediction))

    def run(self):
        self.prepare_data()
        self.predict_prompt_svm()
        self.evaluation_metrics()


if __name__ == "__main__":
    config = yaml.safe_load(open('config/config.yaml'))
    df = pd.read_csv(config['data']['train_features'])
    # TODO: WAS TESTING WITH TRAIN DATA JUST TO SEE IF CODE WORKS, SWITCH TO TEST DATASET
    
    predict = Predictor(df)
    predict.run()


