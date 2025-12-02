import torch
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# --- CONFIGURATION ---
BASE_MODEL = "microsoft/codebert-base"
BIGVUL_ADAPTER_PATH = "./model-bigvul-lora"
JULIET_ADAPTER_PATH = "./model-juliet-lora"
BATCH_SIZE = 32

# --- METRICS ---
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- EVALUATION FUNCTION ---
def evaluate_adapter(adapter_path, dataset, dataset_name):
    print(f"\n========================================================")
    print(f" EVALUATING MODEL: {adapter_path}")
    print(f" ON DATASET:       {dataset_name}")
    print(f"========================================================")
    
    # 1. Load Tokenizer
    try:
        tokenizer = RobertaTokenizer.from_pretrained(adapter_path)
    except:
        print("Warning: Could not load tokenizer from adapter path. Using base tokenizer.")
        tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL)

    # 2. Tokenize Dataset
    print(f"Tokenizing {dataset_name}...")
    def tokenize_function(examples):
        return tokenizer(examples["code"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # Remove raw text column to prevent errors
    tokenized_dataset = tokenized_dataset.remove_columns(["code"])
    tokenized_dataset.set_format("torch")

    # 3. Load Model
    print("Loading Model...")
    base_model = RobertaForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 4. Run Evaluation
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./temp_eval_output", 
            per_device_eval_batch_size=BATCH_SIZE,
            remove_unused_columns=False,
            # *** CRITICAL FIX FOR WINDOWS ***
            dataloader_num_workers=0,  
        ),
        compute_metrics=compute_metrics
    )
    
    print("Running Prediction...")
    result = trainer.predict(tokenized_dataset)
    
    print("\n--- RESULTS ---")
    metrics = result.metrics
    print(f"Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['test_precision']:.4f}")
    print(f"Recall:    {metrics['test_recall']:.4f}")
    print(f"F1 Score:  {metrics['test_f1']:.4f}")
    print("--------------------------------------------------------\n")
    return metrics

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. LOAD DATASETS
    print("--- Loading Datasets ---")

    # A. Big-Vul Test Set
    print("Loading Big-Vul Test Set...")
    bigvul_dataset = load_dataset("bstee615/bigvul", "default", split="test")
    bigvul_dataset = bigvul_dataset.rename_column("func_before", "code")
    bigvul_dataset = bigvul_dataset.rename_column("vul", "labels")
    # Clean columns
    keep_cols = ['code', 'labels']
    remove_cols = [c for c in bigvul_dataset.column_names if c not in keep_cols]
    bigvul_test = bigvul_dataset.remove_columns(remove_cols)

    # B. Juliet Test Set
    print("Loading Juliet Test Set...")
    juliet_dataset = load_dataset("LorenzH/juliet_test_suite_c_1_3", "default", split="test")
    
    # Flatten Juliet
    def restructure_juliet(examples):
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

    juliet_test = juliet_dataset.map(restructure_juliet, batched=True, remove_columns=juliet_dataset.column_names)

    # 2. RUN EXPERIMENTS
    # Experiment A: Big-Vul Model -> Juliet Data
    if os.path.exists(BIGVUL_ADAPTER_PATH):
        evaluate_adapter(BIGVUL_ADAPTER_PATH, juliet_test, "Juliet (Synthetic)")
    
    # Experiment B: Juliet Model -> Big-Vul Data
    if os.path.exists(JULIET_ADAPTER_PATH):
        evaluate_adapter(JULIET_ADAPTER_PATH, bigvul_test, "Big-Vul (Real-World)")