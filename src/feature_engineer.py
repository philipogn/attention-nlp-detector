from utils import analyze_prompt

def extract_features(data, tokenizer, model):
    entropy_scores = []
    variance_scores = []
    for prompt in data['prompt']:
        entropy_score, variance_score = analyze_prompt(prompt, tokenizer, model)
        entropy_scores.append(entropy_score)
        variance_scores.append(variance_score)
        label = data.loc[data['prompt'] == prompt, 'label'].values[0]
        print(f"{prompt[:30]}... | Label: {label} | Entropy: {entropy_score:.4f} | Variance: {variance_score:.4f}")
    data["entropy"] = entropy_scores
    data["variance"] = variance_scores
    return data