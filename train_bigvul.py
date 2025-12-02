import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType

# --- 1. CONFIGURATION ---
MODEL_NAME = "microsoft/codebert-base"
BATCH_SIZE = 16  # Increased from 8 (LoRA uses less memory!)
LEARNING_RATE = 2e-4 # LoRA usually needs a slightly higher LR than full fine-tuning
EPOCHS = 1

# --- 2. LOAD DATASETS (from cache) ---
print("Loading datasets from cache...")
bigvul_dataset = load_dataset("bstee615/bigvul", "default")

# --- 3. PREPARE DATASET (The Fix from before) ---
print("Renaming 'func_before' column to 'code'...")
bigvul_dataset = bigvul_dataset.rename_column("func_before", "code")

# Load Tokenizer
print(f"Loading tokenizer for '{MODEL_NAME}'...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    # We truncate to 512. 
    return tokenizer(
        examples["code"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

print("Tokenizing Big-Vul dataset (this part still takes a moment)...")
# Process training set
tokenized_bigvul_train = bigvul_dataset["train"].map(tokenize_function, batched=True)
# Process validation set
tokenized_bigvul_val = bigvul_dataset["validation"].map(tokenize_function, batched=True)

# Clean up columns
columns_to_remove = [
    'code', 'CVE ID', 'CVE Page', 'CWE ID', 'codeLink', 'commit_id', 
    'commit_message', 'func_after', 'lang', 'project'
]
tokenized_bigvul_train = tokenized_bigvul_train.remove_columns(columns_to_remove)
tokenized_bigvul_val = tokenized_bigvul_val.remove_columns(columns_to_remove)

# Rename labels
print("Renaming 'vul' column to 'labels'...")
tokenized_bigvul_train = tokenized_bigvul_train.rename_column("vul", "labels")
tokenized_bigvul_val = tokenized_bigvul_val.rename_column("vul", "labels")

# Set format
tokenized_bigvul_train.set_format("torch")
tokenized_bigvul_val.set_format("torch")


# --- 4. LOAD MODEL & APPLY LoRA ---
print(f"Loading base model '{MODEL_NAME}'...")
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Define LoRA Config
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # Sequence Classification
    inference_mode=False,
    r=8,                        # Rank (Size of adapters). 8 is standard.
    lora_alpha=32,              # Alpha scaling
    lora_dropout=0.1,
    target_modules=["query", "value"], # Apply LoRA to attention layers
    modules_to_save=["classifier"]
)

# Wrap the model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 
# ^ This will print how many params we are training. It should be < 1%.

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# --- 5. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir="./results-bigvul-lora",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir="./logs-bigvul-lora",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    fp16=True,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_bigvul_train,
    eval_dataset=tokenized_bigvul_val,
)

print("--- Starting Fast Fine-Tuning (LoRA + FP16)... ---")
trainer.train()

print("--- Training Complete ---")

# Save the adapter model
model.save_pretrained("./model-bigvul-lora")
tokenizer.save_pretrained("./model-bigvul-lora")
print("--- LoRA adapters saved to ./model-bigvul-lora ---")