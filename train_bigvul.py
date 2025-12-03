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
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# --- Configuration ---
MODEL_NAME = "microsoft/codebert-base"
BATCH_SIZE = 24
LEARNING_RATE = 2e-4
EPOCHS = 1

# --- Load Datasets (from cache) ---
print("Loading datasets from cache...")
bigvul_dataset = load_dataset("bstee615/bigvul", "default")

# --- Prepare Dataset ---
print("Renaming 'func_before' column to 'code'...")
bigvul_dataset = bigvul_dataset.rename_column("func_before", "code")

# Load Tokenizer
print(f"Loading tokenizer for '{MODEL_NAME}'...")
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
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

# --- Caclulate Class Weights ---
print("Calculating class weights for imbalanced dataset...")
labels = tokenized_bigvul_train["labels"]
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

# Wrap the model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 

# Move to GPU
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

# --- Defining Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results-bigvul-lora",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_dir="./logs-bigvul-lora",
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
    train_dataset=tokenized_bigvul_train,
    eval_dataset=tokenized_bigvul_val,
    class_weights=class_weights,
)

# --- Actually Training the Model ---
print("--- Starting Fast Fine-Tuning (LoRA + FP16)... ---")
trainer.train()

print("--- Training Complete ---")

# Save the adapter model
model.save_pretrained("./model-bigvul-lora")
tokenizer.save_pretrained("./model-bigvul-lora")
print("--- LoRA adapters saved to ./model-bigvul-lora ---")