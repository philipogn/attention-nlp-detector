import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import analyze_prompt

DATASET_TEST = [
    "data/test/deepset_test.csv",
    # "data/test/promptshield_dataset.csv",
    # "data/test/jackhhao_jailbreak_dataset.csv",
    # "data/test/wild_awesome_gen_test.csv"
]

DATASET_TRAIN = [
    "data/train/guychuk_data5000.csv",
    "data/train/harelix_dataset.csv",
    "data/train/GPT_generated_dataset.csv",
    "data/train/deepset_train.csv"
]

def alternate_labels(df): # alternate for better visualisation on dataset
    df0 = df[df['label'] == 0].sample(frac=1, random_state=72).reset_index(drop=True)
    df1 = df[df['label'] == 1].sample(frac=1, random_state=72).reset_index(drop=True)
    min_count = min(len(df0), len(df1)) # use the minimum count to ensure balanced groups (in case counts differ)
    df0 = df0.iloc[:min_count]
    df1 = df1.iloc[:min_count]
    combined = pd.concat([df0, df1], ignore_index=True) # concatenate the two groups

    # create an array of indices that interleaves rows from each group
    n = min_count  # number of rows in each group
    indices = np.empty(2 * n, dtype=int)
    indices[0::2] = np.arange(n)         # even positions: rows from df0 (label 0)
    indices[1::2] = np.arange(n, 2 * n)    # odd positions: rows from df1 (label 1)
    df_alternating = combined.iloc[indices].reset_index(drop=True) # reorder the df based on the interleaved indices
    return df_alternating

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True, return_dict_in_generate=True, attn_implementation="eager", device_map="auto"
    )
    return tokenizer, model

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

def compute_prompt_stats(df, tokenizer, model):
    avg_entropies, avg_variances = [], []
    label0_indices, label1_indices = [], []
    entropy_score_0, entropy_score_1 = [], []
    variance_score_0, variance_score_1 = [], []

    for i, row in df.iterrows():
        prompt, label = row['prompt'], row['label']
        avg_entropy, avg_variance = analyze_prompt(prompt, tokenizer, model)
        avg_entropies.append(avg_entropy)
        avg_variances.append(avg_variance)
        print(f"Prompt: {prompt[:30]} | Label: {label} | Entropy: {avg_entropy:.4f} | Variance: {avg_variance:.4f}")
        if label == 0:
            label0_indices.append(i)
            entropy_score_0.append(avg_entropy)
            variance_score_0.append(avg_variance)
        else:
            label1_indices.append(i)
            entropy_score_1.append(avg_entropy)
            variance_score_1.append(avg_variance)

    print(f"Avg Entropy for | Label 0: {np.mean(entropy_score_0):.4f} | Label 1: {np.mean(entropy_score_1):.4f} | Overall: {np.mean(avg_entropies):.4f}")
    print(f"Avg Variance for | Label 0: {np.mean(variance_score_0):.4f} | Label 1: {np.mean(variance_score_1):.4f} | Overall: {np.mean(avg_variances):.4f}")

    return {"avg_entropies": avg_entropies, "avg_variances": avg_variances, "label0_indices": label0_indices, "label1_indices": label1_indices,
    "entropy_score_0": entropy_score_0, "entropy_score_1": entropy_score_1, "variance_score_0": variance_score_0, "variance_score_1": variance_score_1
    }

def print_entropy_line_graph(stats, model_name):
    plt.figure(figsize=(7, 5))
    plt.plot(stats['label0_indices'], stats['entropy_score_0'], label='Benign prompts', alpha=0.9, linewidth=0.8)
    plt.plot(stats['label1_indices'], stats['entropy_score_1'], label='Malicious prompts', alpha=0.9, linewidth=0.8)
    plt.xlabel("Prompt Index", size=12)
    plt.ylabel("Entropy Score", size=12)
    plt.title(f"Entropy Score by Prompt Index in {model_name}")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.margins(x=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("entropy_graph.png", dpi=300, bbox_inches='tight')
    plt.show()

def print_variance_line_graph(stats, model_name):
    plt.figure(figsize=(7, 5))
    plt.plot(stats['label0_indices'], stats['variance_score_0'], label='Benign prompts', alpha=0.9, linewidth=0.8)
    plt.plot(stats['label1_indices'], stats['variance_score_1'], label='Malicious prompts', alpha=0.9, linewidth=0.8)
    plt.xlabel("Prompt Index", size=12)
    plt.ylabel("Variance Score", size=12)
    plt.title(f"Variance Score by Prompt Index in {model_name}")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.margins(x=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("variance_graph.png", dpi=300, bbox_inches='tight')
    plt.show()

def pca_tfidf(df):
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(df['prompt'])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_tfidf.toarray())

    unique_labels = df['label'].unique()
    colors = ['blue', 'red']

    plt.figure(figsize=(12, 8))
    for label, color in zip(unique_labels, colors):
        indices = df['label'] == label  # get the indices where the label matches
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], c=color, s=10, label=label, alpha=0.9)
    plt.title('PCA of TF-IDF Matrix')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.5)   # light grid
    plt.legend(title="Labels", fontsize=10)
    plt.savefig("pca_tfidf.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    selected_model = "llama"  # change to load saved language models: llama, qwen, phi

    model_options = {
        "llama": "meta-llama/Llama-3.2-1B",
        "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
        "phi": "microsoft/Phi-3-mini-128k-instruct"
    }

    model_name = model_options[selected_model]
    tokenizer, model = load_model_and_tokenizer(model_name)

    df = load_datasets(DATASET_TEST)
    # df = alternate_labels(df)
    pca_tfidf(df)
    
    stats = compute_prompt_stats(df, tokenizer, model)
    
    print_entropy_line_graph(stats, model_name)
    print_variance_line_graph(stats, model_name)