import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType

# --- 1. CONFIGURATION ---
MODEL_NAME = "microsoft/codebert-base"
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 1

# --- 2. LOAD DATASETS (from cache) ---
print("Loading Juliet dataset from cache...")
juliet_dataset = load_dataset("LorenzH/juliet_test_suite_c_1_3", "default")

# --- 3. RESTRUCTURE DATASET ---
print("Restructuring Juliet dataset (flattening 'good' and 'bad' columns)...")

def restructure_function(examples):
    new_codes = []
    new_labels = []
    
    goods = examples['good']
    bads = examples['bad']
    
    for good_code, bad_code in zip(goods, bads):
        if good_code:
            new_codes.append(good_code)
            new_labels.append(0)
        if bad_code:
            new_codes.append(bad_code)
            new_labels.append(1)
            
    return {"code": new_codes, "labels": new_labels}

original_columns = juliet_dataset["train"].column_names
juliet_dataset = juliet_dataset.map(
    restructure_function, 
    batched=True, 
    remove_columns=original_columns
)

print(f"Dataset restructured. New columns: {juliet_dataset['train'].column_names}")

# --- 4. CREATE VALIDATION SPLIT (The Fix) ---
# We split the TRAIN dataset: 90% for training, 10% for validation.
# This keeps the official 'test' split pure and unseen.
print("Splitting training set into Train (90%) and Validation (10%)...")
train_val_split = juliet_dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset_train = train_val_split["train"]
dataset_val = train_val_split["test"]

print(f"Training samples: {len(dataset_train)}")
print(f"Validation samples: {len(dataset_val)}")


# --- 5. PREPARE TOKENIZER ---
print(f"Loading tokenizer for '{MODEL_NAME}'...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["code"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

print("Tokenizing datasets...")
tokenized_juliet_train = dataset_train.map(tokenize_function, batched=True)
tokenized_juliet_val = dataset_val.map(tokenize_function, batched=True)

# Remove the text 'code' column
tokenized_juliet_train = tokenized_juliet_train.remove_columns(["code"])
tokenized_juliet_val = tokenized_juliet_val.remove_columns(["code"])

# Set format
tokenized_juliet_train.set_format("torch")
tokenized_juliet_val.set_format("torch")


# --- 6. LOAD MODEL & APPLY LoRA ---
print(f"Loading base model '{MODEL_NAME}'...")
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# --- 7. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir="./results-juliet-lora",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir="./logs-juliet-lora",
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
    train_dataset=tokenized_juliet_train,
    eval_dataset=tokenized_juliet_val,
)

print("--- Starting Fast Fine-Tuning on Juliet (LoRA + FP16)... ---")
trainer.train()

print("--- Training Complete ---")

# Save the adapter model
model.save_pretrained("./model-juliet-lora")
tokenizer.save_pretrained("./model-juliet-lora")
print("--- LoRA adapters saved to ./model-juliet-lora ---")