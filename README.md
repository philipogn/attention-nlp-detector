## attention-nlp-detector

# Prerequisites for LLaMa 3.2 1B model
The LLaMa model will require access to HuggingFace hub as its a gated model

For access to the LLaMa 3.2 1B model, a HuggingFace account and a generated User Access Token are required to access this gated model.
More information from this link: https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login

1. Create an account on HuggingFace and request access to the LLaMa 3.2 1B model:
https://huggingface.co/meta-llama/Llama-3.2-1B

2. Install the HuggingFace Hub CLI with the command:
```sh
pip install -U "huggingface_hub[cli]"
```

3. Generate a User Access Token from Account Setting Page, and log in:
```sh
huggingface-cli login
```

# How to install
1. Clone the repository
2. 