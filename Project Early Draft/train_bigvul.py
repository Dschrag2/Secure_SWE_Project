import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)

# --- 1. LOAD DATASETS (from cache) ---
print("Loading datasets from cache...")
bigvul_dataset = load_dataset("bstee615/bigvul", "default")


# --- 2. LOAD TOKENIZER AND MODEL (from cache) ---
model_name = "microsoft/codebert-base"
print(f"Loading tokenizer and model for '{model_name}'...")

tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"--- Model loaded and moved to {device} ---")


# --- 3. PREPARE BIG-VUL DATASET ---

# Rename 'func_before' to 'code'
print("Renaming 'func_before' column to 'code'...")
bigvul_dataset = bigvul_dataset.rename_column("func_before", "code")

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["code"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

print("Tokenizing Big-Vul dataset...")
tokenized_bigvul_train = bigvul_dataset["train"].map(tokenize_function, batched=True)
tokenized_bigvul_val = bigvul_dataset["validation"].map(tokenize_function, batched=True)

# Remove columns I don't need to save memory
columns_to_remove = [
    'code', 'CVE ID', 'CVE Page', 'CWE ID', 'codeLink', 'commit_id', 
    'commit_message', 'func_after', 'lang', 'project'
]
tokenized_bigvul_train = tokenized_bigvul_train.remove_columns(columns_to_remove)
tokenized_bigvul_val = tokenized_bigvul_val.remove_columns(columns_to_remove)

# Rename the 'vul' column to 'labels'
print("Renaming 'vul' column to 'labels'...")
tokenized_bigvul_train = tokenized_bigvul_train.rename_column("vul", "labels")
tokenized_bigvul_val = tokenized_bigvul_val.rename_column("vul", "labels")

# Set the format to torch tensors
tokenized_bigvul_train.set_format("torch")
tokenized_bigvul_val.set_format("torch")

print("--- Tokenization Complete ---")


# --- 4. SET UP AND RUN THE TRAINER ---

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results-bigvul",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs-bigvul",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
)

# Create the Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_bigvul_train,
    eval_dataset=tokenized_bigvul_val,
)

print("--- Starting Fine-Tuning on Big-Vul... ---")
trainer.train()

print("--- Training Complete ---")

trainer.save_model("./model-trained-on-bigvul")
tokenizer.save_pretrained("./model-trained-on-bigvul")
print("--- Best model saved to ./model-trained-on-bigvul ---")