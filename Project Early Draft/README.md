# Secure_SWE_Project
Secure Software Engineering Project: Fine-tuning large language models for software vulnerability detection

## Datasets used:
- Big-Vul: https://huggingface.co/datasets/bstee615/bigvul
- Juliet Test Suite: https://huggingface.co/datasets/LorenzH/juliet_test_suite_c_1_3

## Model used:
- Microsoft codeBERT-base

## Code
- train_bigvul.py: This is a python script that I can run to fine-tune the CodeBERT-Base model on the Big-Vul Dataset. This code was originally done in a Jupyter notebook, but was converted to a script for ease of fine-tuning. This is currently the only script I have, but it will be fairly straightforward to copy this code to fine-tune on the Juliet training set.
