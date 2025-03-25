# Fine-tuning DistilBERT on the CoLA Dataset

## Overview
This project demonstrates how to fine-tune **DistilBERT** on the **CoLA (Corpus of Linguistic Acceptability)** dataset using Hugging Face's **Transformers** and **Datasets** libraries. The goal is to classify sentences as grammatically acceptable or unacceptable. The notebook explores two fine-tuning approaches: **Full Parameter Fine-Tuning** and **Transfer Learning with Additional Layers**.

## Dataset
- **Dataset**: [CoLA](https://gluebenchmark.com/tasks)
- **Task**: Binary classification (Acceptable vs. Unacceptable sentences)
- **Source**: Part of the GLUE Benchmark
- **Format**: Train, validation, and test splits

## Fine-Tuning Approaches
### 1. Full Parameter Fine-Tuning
- Fine-tunes **all layers** of the pre-trained DistilBERT model.
- Uses **AdamW optimizer**, **cross-entropy loss**, and **learning rate scheduling**.
- Achieves **higher accuracy** but requires more training time and computational resources.

### 2. Transfer Learning with Additional Layers
- **Freezes** the DistilBERT base model and adds:
  - **Batch Normalization**
  - **Dropout**
  - **Fully Connected Layers** for classification.
- Trains only the newly added layers, reducing computation time.
- Can help prevent overfitting on smaller datasets.

## Steps in the Notebook
1. **Environment Setup**
   - Install necessary libraries (`transformers`, `datasets`, `torch`)
   - Import dependencies

2. **Load and Explore the Dataset**
   - Load CoLA dataset using Hugging Face's `datasets` library
   - Perform exploratory data analysis (EDA) with Pandas & Seaborn

3. **Data Preprocessing**
   - Tokenization using `DistilBertTokenizer`
   - Sentence length and stopword analysis
   - Convert dataset into tokenized format

4. **Model Training**
   - **Approach 1: Full Parameter Fine-Tuning**
     - Train entire **DistilBERT** (`distilbert-base-uncased`)
     - Optimized with **AdamW**
   - **Approach 2: Transfer Learning with Custom Layers**
     - Freeze base model, train only additional layers
     - Uses **Dropout, BatchNorm, and FC layers**

5. **Evaluation**
   - Compute **Accuracy, F1-score, MCC (Matthews Correlation Coefficient)**
   - Generate **Confusion Matrix**
   - Compare **baseline vs. fine-tuned model performance**

6. **Results Comparison**

| Model | Accuracy | F1 Score | MCC |
|--------|----------|---------|------|
| Baseline | 0.34 | 0.25 | 0.02 |
| Full Fine-Tuned DistilBERT | 0.79 | 0.78 | 0.48 |
| Transfer Learning with Custom Layers | 0.81 | 0.80 | 0.53 |


## Deployment
The fine-tuned model can be deployed on **Hugging Face Model Hub**:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model.save_pretrained("your-huggingface-model-name")
tokenizer.save_pretrained("your-huggingface-model-name")
```

Alternatively, you can use **FastAPI** or **Gradio** for deployment.

## How to Use
1. Clone this repository:
   ```sh
   git clone https://github.com/your-repo/fine-tuning-distilbert-cola.git
   cd fine-tuning-distilbert-cola
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the notebook or script:
   ```sh
   jupyter notebook FinetuningDistilBERT.ipynb
   ```
4. Train the model and evaluate results.

## References
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [GLUE Benchmark](https://gluebenchmark.com/)
- [CoLA Dataset](https://nyu-mll.github.io/CoLA/)

