# attention-nlp-detector

## Prerequisites for LLaMa 3.2 1B model
For access to the LLaMa 3.2 1B model, a HuggingFace account and a generated User Access Token are required to access this gated model.
More information from this link: https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login

1. Create an account on HuggingFace and request access to the LLaMa 3.2 1B model:
https://huggingface.co/meta-llama/Llama-3.2-1B

2. Install the HuggingFace Hub CLI with the command:
```sh
pip install -U "huggingface_hub[cli]"
```

3. Generate a User Access Token from Account Setting Page, and use the token to log in:
```sh
huggingface-cli login
```

# How to install and run
1. Clone the repository
2. Install the required dependencies by running the command:
```sh
pip install -r requirements.txt
```
3. The main detector file can then be run from the root directory of the repository with:
```sh
python interactive_detector.py
```

