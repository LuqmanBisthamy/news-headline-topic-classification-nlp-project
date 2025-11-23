# 📰 News Headline Topic Classification — NLP Group Project

This repository contains a complete Natural Language Processing (NLP) workflow for classifying news headlines into predefined categories using both classical machine learning techniques and a fine-tuned BERT transformer model.

The project compares **TF-IDF + SVM, Logistic Regression, Random Forest** against a **fine-tuned BERT model** to evaluate which method performs best on short text classification.

---

## ⭐ Project Overview
This project aims to build an automated system capable of classifying news headlines into topics.  
The study evaluates:

- **Classical ML models**
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest  
- **Deep Learning**
  - Fine-tuned BERT (Bidirectional Encoder Representations from Transformers)

The experiment determines whether traditional vectorization (TF-IDF) or transformer-based embeddings yield better performance.

---

## 📁 Files in This Repository

| File | Description |
|------|-------------|
| `Topic Classification Of News Headlines Using TF-IDF with SVM, LR, RF, and Fine-Tuned BERT(Code).ipynb` | Full Jupyter Notebook containing preprocessing, feature extraction, model training, and evaluation |
| `train.csv` / `test.csv` | Labeled dataset used for training and evaluation |

---

## 🧬 Dataset
The original training dataset is **not included in this repository** because of GitHub’s file size limits.

📌 You can download the full training dataset from Kaggle: 
👉 https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset/data

Each sample includes:
- **text** → the headline  
- **category** → topic label (target)
 
Test splits are provided as `test.csv`.

---

## 🛠️ Methodology

### 1️⃣ Data Preprocessing
- Lowercasing  
- Removing punctuation  
- Stopword removal  
- Tokenization  
- Lemmatization  

---

### 2️⃣ Feature Engineering
#### **TF-IDF Vectorization**
Transforms raw text into weighted numerical vectors suitable for classical ML algorithms.

#### **BERT Tokenization**
- WordPiece tokenizer  
- Adds `[CLS]`, `[SEP]` tokens  
- Pads/truncates sequences to a max length  
- Converts tokens → embeddings  

---

### 3️⃣ Trained Models

#### **Classical Machine Learning**
- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  

These models are trained on TF-IDF vectors.

---

#### **Fine-Tuned BERT**
- Based on `bert-base-uncased`  
- Additional classification head  
- Fine-tuned on training dataset  
- Trained using:
  - AdamW optimizer  
  - Cross-entropy loss  
  - Learning rate scheduling  
  - 2–4 epochs  

---

## 📈 Evaluation Metrics
Models are evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- Classification Report  

---

## 🥇 Model Comparison (Summary)

| Model | Performance Summary |
|-------|---------------------|
| Logistic Regression | Works decently on TF-IDF but inconsistent on certain topics |
| Random Forest | Underperforms due to sparse vector input |
| **SVM (TF-IDF)** | Best among classical ML models |
| **BERT (Fine-tuned)** | **Highest accuracy, precision, recall, and F1-score** |

### 🔥 Final Winner: **Fine-Tuned BERT**  
Semantic understanding gives BERT a significant advantage over TF-IDF-based classical models.

---

## 🧠 Key Findings
- Classical ML models depend on surface-level word patterns, limiting accuracy.  
- BERT captures contextual meaning → performs best on short sentences.  
- SVM is the strongest baseline for TF-IDF text classification.  
- Fine-tuned BERT significantly improves performance across all metrics.

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Launch Jupyter Notebook
```bash
jupyter notebook
```

### 3️⃣ Open and run the notebook
```bash
Topic Classification Of News Headlines Using TF-IDF with SVM, LR, RF, and Fine-Tuned BERT(Code).ipynb
```

---

## 🧩 Project Workflow Diagram

<p align="center">

  <strong>[ Data Loading ]</strong>  
  ↓  
  <strong>[ Text Cleaning ]</strong>  
  ↓  
  <strong>[ TF-IDF / BERT Tokenization ]</strong>  
  ↓  
  <strong>[ Model Training ]</strong>  
  ↓  
  <strong>[ Evaluation & Comparison ]</strong>

</p>

---

## 👤 My Contribution
This project highlights your skills in:

- Natural Language Processing
- Classical ML + Deep Learning
- TF-IDF feature engineering
- Transformer fine-tuning
- Model evaluation & analysis
- Python, Scikit-Learn, PyTorch

- ---
