# Fine-tuning-BERT-on-CoLa-Dataset

# Fine-tuning DistilBERT on the CoLA Dataset

## Overview
This project demonstrates how to fine-tune **DistilBERT** on the **CoLA (Corpus of Linguistic Acceptability)** dataset using Hugging Face's **Transformers** and **Datasets** libraries. The goal is to classify sentences as grammatically acceptable or unacceptable. The notebook includes data preprocessing, model training, and evaluation.

## Dataset
- **Dataset**: [CoLA](https://gluebenchmark.com/tasks)
- **Task**: Binary classification (Acceptable vs. Unacceptable sentences)
- **Source**: Part of the GLUE Benchmark
- **Format**: Train, validation, and test splits

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
   - Load pre-trained **DistilBERT** (`distilbert-base-uncased`)
   - Fine-tune using **Trainer API** with:
     - Cross-entropy loss
     - AdamW optimizer
     - Learning rate scheduler
   - Train on GPU (if available)

5. **Evaluation**
   - Compute **Accuracy, F1-score, MCC (Matthews Correlation Coefficient)**
   - Generate **Confusion Matrix**
   - Compare **baseline vs. fine-tuned model performance**

6. **Custom Model Enhancement**
   - Implement additional layers: **Pre-classification, Batch Normalization, Dropout**
   - Fine-tune enhanced model and compare performance

## Results
| Model | Accuracy | F1 Score | MCC |
|--------|----------|---------|------|
| Baseline | 0.34 | 0.25 | 0.02 |
| Fine-tuned DistilBERT | 0.79 | 0.78 | 0.48 |
| Enhanced DistilBERT | 0.81 | 0.80 | 0.53 |
