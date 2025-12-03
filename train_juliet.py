import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# --- Configuration ---
MODEL_NAME = "microsoft/codebert-base"
BATCH_SIZE = 24
LEARNING_RATE = 2e-4
EPOCHS = 1

# --- Load Datasets (from cache) ---
print("Loading Juliet dataset from cache...")
juliet_dataset = load_dataset("LorenzH/juliet_test_suite_c_1_3", "default")

# --- Restructure Data ---
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

# --- Create Validation Set ---
# Split the TRAIN dataset: 90% for training, 10% for validation.
print("Splitting training set into Train (90%) and Validation (10%)...")
train_val_split = juliet_dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset_train = train_val_split["train"]
dataset_val = train_val_split["test"]

print(f"Training samples: {len(dataset_train)}")
print(f"Validation samples: {len(dataset_val)}")


# --- Prepare Tokenizer ---
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

# --- Calculate Class Weights ---
print("Calculating class weights for dataset...")
labels = tokenized_juliet_train["labels"]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=np.array(labels)
)
print(f"Class weights: [Non-vulnerable: {class_weights[0]:.4f}, Vulnerable: {class_weights[1]:.4f}]")

# --- Load Model and apply LoRA ---
print(f"Loading base model '{MODEL_NAME}'...")
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Define LoRA Config
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    modules_to_save=["classifier"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Send model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# --- Training with Class Weights ---
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Move class weights to the same device as logits
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float, device=logits.device)
        else:
            weight = None

        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results-juliet-lora",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir="./logs-juliet-lora",
    logging_steps=200,
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    fp16=True,
    load_best_model_at_end=True,
    dataloader_num_workers=0,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_juliet_train,
    eval_dataset=tokenized_juliet_val,
    class_weights=class_weights,
)

# Actually training the model
print("--- Starting Fast Fine-Tuning on Juliet (LoRA + FP16)... ---")
trainer.train()

print("--- Training Complete ---")

# Save the adapter model
model.save_pretrained("./model-juliet-lora")
tokenizer.save_pretrained("./model-juliet-lora")
print("--- LoRA adapters saved to ./model-juliet-lora ---")