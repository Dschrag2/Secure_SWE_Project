# Synthetic vs. Real: Cross-Dataset Vulnerability Detection with CodeBERT

A research project exploring cross-dataset transfer learning between synthetic (Juliet Test Suite) and real-world (BigVul) vulnerability datasets using parameter-efficient fine-tuning (LoRA) and class-weighted loss functions.

---

## 📊 Project Overview

This project investigates how well vulnerability detection models transfer across different datasets by:
- Fine-tuning CodeBERT on BigVul (real-world CVE data) and testing on Juliet (synthetic test cases)
- Fine-tuning CodeBERT on Juliet and testing on BigVul
- Implementing **class-weighted loss functions** to handle severe class imbalance (94.2% non-vulnerable in BigVul)
- Using LoRA for parameter-efficient fine-tuning (~0.7% trainable parameters)

### Key Results

| Experiment | Accuracy | Precision | Recall | F1 Score |
|------------|----------|-----------|--------|----------|
| **BigVul → Juliet** | 59.1% | **74.3%** | 27.8% | **40.5%** |
| **Juliet → BigVul** | 92.2% | 7.7% | 13.7% | 9.8% |

---

## 🗂️ Repository Structure

```
├── train_bigvul.py       # Fine-tune CodeBERT on BigVul dataset
├── train_juliet.py       # Fine-tune CodeBERT on Juliet Test Suite
├── Evaluate.py           # Cross-dataset evaluation script
├── requirements.txt      # Python dependencies
│
├── model-bigvul-lora/    # Trained LoRA adapters (BigVul)
├── model-juliet-lora/    # Trained LoRA adapters (Juliet)
├── results-bigvul-lora/  # Training checkpoints (BigVul)
├── results-juliet-lora/  # Training checkpoints (Juliet)
└── temp_eval_output/     # Evaluation results
```

---

## 🔬 Methodology

### Base Model
- **Microsoft CodeBERT** (`microsoft/codebert-base`)
- Pre-trained on code and natural language

### Datasets

| Dataset | Type | Size | Class Balance | Source |
|---------|------|------|---------------|--------|
| **BigVul** | Real-world CVEs | ~150K | 94.2% non-vulnerable | [HuggingFace](https://huggingface.co/datasets/bstee615/bigvul) |
| **Juliet Test Suite** | Synthetic CWE cases | ~80K | 50/50 balanced | [HuggingFace](https://huggingface.co/datasets/LorenzH/juliet_test_suite_c_1_3) |

### Training Configuration
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
  - Rank: 8, Alpha: 32, Dropout: 0.1
  - Target modules: Query & Value attention layers
  - Trainable parameters: **0.7%** of total model
- **Optimization:**
  - Learning rate: 2e-4
  - Batch size: 24
  - FP16 mixed precision
  - Epochs: 1
- **Class Weights:**
  - BigVul: `[0.53, 8.66]` (16× penalty for missing vulnerabilities)
  - Juliet: `[1.00, 1.00]` (balanced)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: 6GB+ VRAM)
- CUDA Toolkit (if training on GPU)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Secure_SWE_Project.git
   cd Secure_SWE_Project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv sec-env

   # On Windows:
   sec-env\Scripts\activate

   # On Linux/Mac:
   source sec-env/bin/activate
   ```

3. **Install PyTorch with CUDA** (if using GPU)
   ```bash
   # Example for CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### Training

**Train on BigVul dataset:**
```bash
python train_bigvul.py
```
- Training time: ~2.5-4 hours (RTX 3060 Laptop)
- Output: `./model-bigvul-lora/`

**Train on Juliet Test Suite:**
```bash
python train_juliet.py
```
- Training time: ~3-5 hours (RTX 3060 Laptop)
- Output: `./model-juliet-lora/`

### Evaluation

**Run cross-dataset evaluation:**
```bash
python Evaluate.py
```

This will:
1. Load both trained models
2. Test BigVul model on Juliet test set
3. Test Juliet model on BigVul test set
4. Print metrics: Accuracy, Precision, Recall, F1 Score

---

## 🙏 Acknowledgments

- **Datasets:**
  - BigVul: [bstee615/bigvul](https://huggingface.co/datasets/bstee615/bigvul)
  - Juliet Test Suite: [LorenzH/juliet_test_suite_c_1_3](https://huggingface.co/datasets/LorenzH/juliet_test_suite_c_1_3)
- **Models:**
  - Microsoft CodeBERT: [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
- **Libraries:**
  - Hugging Face Transformers & PEFT
  - PyTorch

---