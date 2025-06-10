# 🎬 IMDB Sentiment Analyzer: TF‑IDF → Logistic Regression Pipeline
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/your-repo-name/blob/main/notebook.ipynb)  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]() [![scikit-learn](https://img.shields.io/badge/scikit--learn-%3E%3D1.0-orange.svg)]() [![NLTK](https://img.shields.io/badge/NLTK-%3E%3D3.7-purple.svg)]() [![Joblib](https://img.shields.io/badge/Joblib-%3E%3D1.0-gray.svg)]() [![Matplotlib](https://img.shields.io/badge/Matplotlib-%3E%3D3.5-red.svg)]() [![Seaborn](https://img.shields.io/badge/Seaborn-%3E%3D0.11-teal.svg)]()
---
## 🚀 Project Overview

## 🛠️ Setup & Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training
```bash
python src/train.py --data-path path/to/IMDBDataset.csv --model-out sentiment_lr_model.joblib --vectorizer-out tfidf_vectorizer.joblib
```

### Prediction
```bash
python src/predict.py --model sentiment_lr_model.joblib --vectorizer tfidf_vectorizer.joblib --text "That movie was phenomenal! Best I've seen all year."
```

## 📈 Sample Results

| Stage | Accuracy |
|-------|----------|
| LogisticRegression (base) | 0.88 |
| LogisticRegression (tuned) | 0.90 |

## ⚠️ Critical Considerations

• **Vocabulary Size:** TF-IDF may blow up memory on rare tokens. Consider trimming max_features or using hashing.  
• **Imbalance:** We use class_weight='balanced', but you might try focal loss or SMOTE for fine-tuning.  
• **Model Choice:** A simple linear model is fast—swap in an SVM or even BERT for higher accuracy.  
• **Preprocessing Sanity:** Always sanity-check on random text or blanks to catch tokenization bugs.

## 🤝 Contributing

1. Fork this repo
2. Create a new branch (`git checkout -b feat/my-feature`)
3. Commit your changes (`git commit -m "Add awesome feature"`)
4. Push (`git push origin feat/my-feature`)
5. Open a Pull Request
