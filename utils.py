import torch
import numpy as np
import gc
import re

def preprocess_text(text): # preprocess function for tfidf
    text = text.lower()
    return text

def get_attention_data(prompt, tokenizer, model):
    device = next(model.parameters()).device  # get model's device
    inputs = tokenizer(prompt, return_tensors="pt") # inputs returned as tensor
    inputs = {k: v.to(device) for k, v in inputs.items()}  # move inputs to model's device
    with torch.no_grad():   # no gradient needed, saves memory
        outputs = model(**inputs)
    torch.cuda.empty_cache()    # free GPU memory after each prompt
    return outputs.attentions

def select_early_middle_layers(model, num_layers_to_select=4):
    num_of_layers = model.config.num_hidden_layers  # get number of layers from llm.
    middle_layer = num_of_layers // 2  # get middle layer.
    selected_layers = list(range(middle_layer - num_layers_to_select, middle_layer)) # select layers from middle layer to 4 previous layers.
    return selected_layers

def get_selected_attention_layers(attentions, model):
    layers = select_early_middle_layers(model)  # call func to select layers
    selected = [attentions[i] for i in layers]  # select only the layers required
    stacked_selected_layers = torch.stack(selected) # stack all selected into single tensor, much smaller
    return stacked_selected_layers

def compute_entropy(attention_tensor):
    entropies = []
    attention_tensor = attention_tensor.squeeze(1)  # remove batch dimension as processing one input, so its redundant
    selected_num_layers, selected_num_heads, _, _ = attention_tensor.shape # get subset of layers and all heads
    for l in range(selected_num_layers):
        for h in range(selected_num_heads):
            head_matrix = attention_tensor[l, h]    # get Q x K matrix for each head  (row x column)
            row_entropies = - (head_matrix * torch.log2(head_matrix + 1e-9)).sum(dim=-1) # compute entropy for each row
            head_entropy = row_entropies.mean() # average entropy across all rows
            entropies.append(head_entropy.item())
    return np.mean(entropies)

def compute_variance(attention_tensor):
    variances = []
    attention_tensor = attention_tensor.squeeze(1)  # remove batch dimension 
    selected_num_layers, selected_num_heads, _, _ = attention_tensor.shape # get subset of layers and all heads
    for l in range(selected_num_layers):
        for h in range(selected_num_heads):
            head_matrix = attention_tensor[l, h]    # get Q x K matrix for each head (row x column)
            row_variance = torch.var(head_matrix, dim=-1)   # compute variance for each row
            head_variance = row_variance.mean() # average variance across all rows
            variances.append(head_variance.item())
    return np.mean(variances)

def analyze_prompt(prompt, tokenizer, model):
    attentions = get_attention_data(prompt, tokenizer, model)
    selected_attention_layers = get_selected_attention_layers(attentions, model)
    entropy_score = compute_entropy(selected_attention_layers)
    variance_score = compute_variance(selected_attention_layers)

    torch.cuda.empty_cache()
    gc.collect()
    return entropy_score, variance_score
