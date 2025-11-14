# Secure_SWE_Project
Secure Software Engineering Project: Fine-tuning large language models for software vulnerability detection

## Datasets used:
- Big-Vul: https://huggingface.co/datasets/bstee615/bigvul
- Juliet Test Suite: https://huggingface.co/datasets/LorenzH/juliet_test_suite_c_1_3

## Model used:
- Microsoft codeBERT-base

## Code
- train_bigvul.py: This is a python script that I can run to fine-tune the CodeBERT-Base model on the Big-Vul Dataset. This code was originally done in a Jupyter notebook, but was converted to a script for ease of fine-tuning. This is currently the only script I have, but it will be fairly straightforward to copy this code to fine-tune on the Juliet training set.

## Project Reframing

I am reframing the project to focus solely on Reserach Question #3:
- What are the Strengths and weaknesses of LLM-based approaches across datasets (synthetic vs. real-world) and vulnerability types?

1. I am going to fine-tune one model on the Big-Vul dataset and test on the Juliet dataset. This will show how well a model trained on messy, real-world bugs can find clean, textbook bugs
2. Next, I am going to fine-tune the same LLM on the Juliet dataset and test on the Big-Vul dataset. This will show how well a model trained on textbook bugs can find real-world bugs

The novelty of this paper is a deep, focused analysis of cross-dataset generalization. This will be the first paper to provide a clear answer on whether these models are just "memorizing" a dataset's quirks or learning the actual concept of a vulnerability.

