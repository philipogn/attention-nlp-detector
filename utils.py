import torch
import numpy as np
import gc
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[\n\r\t]', ' ', text)
    return text

def select_early_middle_layers(model, num_layers_to_select=4):
    num_of_layers = model.config.num_hidden_layers  # get number of layers in the model.
    middle_layer = num_of_layers // 2  # get middle layer.
    selected_layers = list(range(middle_layer - num_layers_to_select, middle_layer)) # select layers from middle layer to 4 previous layers.
    return selected_layers

def get_attention_data(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt")#.to("cpu")
    outputs = model(**inputs)
    torch.cuda.empty_cache()    # free GPU memory after each prompt
    return outputs.attentions

def get_selected_attention(attentions, model):
    layers = select_early_middle_layers(model)
    stacked_attentions = torch.stack(attentions) # stack all attentions into single tensor
    selected_layers = stacked_attentions[layers] # select only the layers required and return
    return selected_layers

def compute_entropy(attention_tensor):
    entropies = []
    attention_tensor = attention_tensor.squeeze(1)  # remove batch dimension
    num_layers, num_heads, _, _ = attention_tensor.shape
    for l in range(num_layers):
        for h in range(num_heads):
            head_matrix = attention_tensor[l, h]    # get Q x K matrix for each head
            row_entropies = - (head_matrix * torch.log2(head_matrix + 1e-9)).sum(dim=-1) # compute entropy for each row
            head_entropy = row_entropies.mean() # average entropy across all rows
            entropies.append(head_entropy.item())
    return np.mean(entropies)

def compute_variance(attention_tensor):
    variances = []
    attention_tensor = attention_tensor.squeeze(1)  # remove batch dimension
    num_layers, num_heads, _, _ = attention_tensor.shape
    for l in range(num_layers):
        for h in range(num_heads):
            head_matrix = attention_tensor[l, h]    # get Q x K matrix for each head
            row_variance = torch.var(head_matrix, dim=-1).mean()
            variances.append(row_variance.item()) # average variance across all rows
    return np.mean(variances)

def analyze_prompt(prompt, tokenizer, model):
    torch.cuda.empty_cache()
    gc.collect()

    attentions = get_attention_data(prompt, tokenizer, model)
    selected_attention = get_selected_attention(attentions, model)
    entropy_score = compute_entropy(selected_attention)
    variance_score = compute_variance(selected_attention)

    torch.cuda.empty_cache()
    gc.collect()
    return entropy_score, variance_score
