import yaml
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import analyze_prompt

class FeatureEngineer():
    def __init__(self, config):
        self.config = config
        self.model_name = config['language_model']['qwen']
        self.train_set = config['data']['train_dataset']
        self.train_df = None
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self):
        parameters = self.config['language_model_parameters']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **parameters)
            # output_attentions=True, return_dict_in_generate=True, attn_implementation="eager", device_map="auto"
        return self.tokenizer, self.model

    def load_datasets(self):
        '''
        Load training datasets and concatenate into a single dataframe
        '''
        dataframes = []
        for path in self.train_set:
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

    def extract_features(self):
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
        return self.train_df

    def feature_to_csv(self):
        if self.train_df is None:
            print('Failed to save features')
        else:
            self.train_df.to_csv('data/2-features/train_set_features.csv')

    def run(self):
        self.load_model_and_tokenizer()
        self.load_datasets()
        self.extract_features()
        self.feature_to_csv()


if __name__ == '__main__':
    config = yaml.safe_load(open('config/config.yaml'))

    features = FeatureEngineer(config)
    features.run()