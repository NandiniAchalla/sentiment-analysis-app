# 💬 Sentiment Classification using VADER and RoBERTa

## 📘 Overview
This project classifies **Amazon product reviews** as *positive, negative, or neutral* using both **traditional lexicon-based NLP** and **modern transformer-based models**.  

It demonstrates how sentiment analysis can evolve from rule-based systems to deep learning transformers, providing a comparison of performance and accuracy between the two methods.

## ⚙️ Tech Stack
- **Programming:** Python  
- **Libraries:** NLTK, VADER, Transformers (HuggingFace), PyTorch, pandas, NumPy, scikit-learn  
- **Environment:** Jupyter Notebook / Google Colab  

## 🧠 Objective
- Preprocess and clean real-world text data (Amazon Reviews).  
- Apply **VADER** for lexicon-based sentiment scoring.  
- Fine-tune **RoBERTa** transformer model for text classification.  
- Compare both models using accuracy, precision, recall, and F1-score.  

## 📊 Dataset
- **Source:** Amazon Reviews dataset (public domain)  
- **Features:**
  - `reviewText` — the text of the customer review  
  - `overall` — rating given by the customer (used to infer sentiment)  

## 🚀 Implementation Steps

### 1. **Data Preprocessing**
- Removed null values and duplicates  
- Tokenized and cleaned text (lowercasing, stopword removal, punctuation handling)  
- Labeled reviews as:
  - Positive (rating ≥ 4)
  - Neutral (rating = 3)
  - Negative (rating ≤ 2)

### 2. **Model 1 — VADER (Rule-Based NLP)**
- Used **VADER** from `nltk.sentiment` for lexicon-based scoring.  
- Classified sentiment using compound scores:
  - `compound > 0.05` → Positive  
  - `compound < -0.05` → Negative  
  - Otherwise → Neutral

### 3. **Model 2 — RoBERTa (Transformer Fine-Tuning)**
- Used **HuggingFace Transformers** library with pretrained `roberta-base` model.  
- Tokenized reviews with `RobertaTokenizer`.  
- Fine-tuned model using PyTorch for 3 epochs on GPU.  
- Evaluated using accuracy and F1-score.

## 📈 Results

**VADER Model:**
Accuracy: 0.79
Precision: 0.78
Recall: 0.77
F1 Score: 0.77

**RoBERTa (Fine-tuned) Model:**
Accuracy: 0.92
Precision: 0.91
Recall: 0.92
F1 Score: 0.92
✅ **RoBERTa outperformed VADER**, showing how transformer models can capture deeper semantic meaning in text.

