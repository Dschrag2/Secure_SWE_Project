import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel

# CONFIG
BASE_MODEL = "microsoft/codebert-base"
ADAPTER_PATH = "./model-juliet-lora" 

print(f"--- Debugging Model: {ADAPTER_PATH} ---")

# 1. Load Tokenizer
tokenizer = RobertaTokenizer.from_pretrained(ADAPTER_PATH)

# 2. Load Model
print("Loading base model...")
base_model = RobertaForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# --- DIAGNOSTIC: CHECK IF CLASSIFIER LOADED ---
print("\n[DIAGNOSTIC] Verifying Classifier Weights...")
# Load a fresh, random model to compare against
fresh_model = RobertaForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
# Get weights from your loaded model (wrapped in PEFT)
# Note: PEFT wraps the base model, so we access classifier via model.base_model.model.classifier
loaded_weights = model.base_model.model.classifier.dense.weight
random_weights = fresh_model.classifier.dense.weight

if torch.equal(loaded_weights, random_weights):
    print("❌ CRITICAL FAILURE: The classifier weights are identical to a random initialization.")
    print("   -> This means your trained classification head was NOT loaded.")
    print("   -> Check if 'modules_to_save' was correctly set during training.")
else:
    print("✅ SUCCESS: Classifier weights are different from random initialization.")
    print("   -> The trained head loaded correctly. If predictions are bad, the model needs more training.")


# 3. Create Inputs (MATCHING TRAINING CONFIG EXACTLY)
safe_code = "int main() { return 0; }"
vuln_code = """
void bad() {
    char buffer[10];
    strcpy(buffer, "This string is definitely too long for the buffer");
}
"""

# Force max_length=512 to match training exactly
inputs_safe = tokenizer(safe_code, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
inputs_vuln = tokenizer(vuln_code, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

# Verify inputs are different
if torch.equal(inputs_safe['input_ids'], inputs_vuln['input_ids']):
    print("\n❌ DATA ERROR: Inputs are identical!")
else:
    print("\n✅ DATA CHECK: Inputs are distinct.")

# 4. Get Predictions
print("\nRunning Inference...")
with torch.no_grad():
    outputs_safe = model(**inputs_safe)
    outputs_vuln = model(**inputs_vuln)

logits_safe = outputs_safe.logits
logits_vuln = outputs_vuln.logits

probs_safe = torch.softmax(logits_safe, dim=-1)
probs_vuln = torch.softmax(logits_vuln, dim=-1)

print("\n--- TEST RESULTS ---")
print(f"Safe Code Logits:    {logits_safe.numpy()[0]}")
print(f"Safe Code Probs:     {probs_safe.numpy()[0]} (Target: Class 0)")
print("-" * 30)
print(f"Vuln Code Logits:    {logits_vuln.numpy()[0]}")
print(f"Vuln Code Probs:     {probs_vuln.numpy()[0]} (Target: Class 1)")

# Interpret
safe_score = probs_safe[0][0].item()
vuln_score = probs_vuln[0][1].item()

if safe_score > 0.5 and vuln_score > 0.5:
    print("\n✅ RESULT: Model successfully distinguishes Safe from Vulnerable!")
elif safe_score > 0.5 and vuln_score < 0.5:
    print("\n⚠️ ISSUE: Model predicts 'Safe' for everything (False Negative).")
elif safe_score < 0.5 and vuln_score > 0.5:
    print("\n⚠️ ISSUE: Model predicts 'Vulnerable' for everything (False Positive).")
else:
    print("\n⚠️ ISSUE: Model is confused or predicting randomly.")