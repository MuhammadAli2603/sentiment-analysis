IMDB Sentiment Analyzer: TF-IDF & Logistic Regression
*License:* MIT
*Stack:* Python · scikit-learn · NLTK · Joblib · Matplotlib · Seaborn
🚀 Project Overview
An end-to-end sentiment-analysis pipeline on the IMDB movie-review dataset, implemented in a single Colab notebook and modularized for local runs:

- Preprocessing: NLTK-powered tokenization, stop-word removal, stemming, and text-cleaning.
- Features: TF-IDF vectorization to capture term importance.
- Model: Logistic Regression with class_weight='balanced' for robust performance.
- Evaluation: Classification report and confusion-matrix visualizations.
- Inference: Simple script to drop in your own review text and get a positive/negative prediction.
📂 Repository Structure
~~~text
imdb-sentiment-analysis/
├── Untitled16.ipynb ← All-in-one Colab demo
├── src/
│   ├── preprocess.py ← Text-cleaning & tokenization
│   ├── train.py ← Train, evaluate, and save model
│   └── predict.py ← Load artifacts & predict on new text
├── requirements.txt ← Pinned dependencies
├── .gitignore ← Excludes venv, caches, data, artifacts
└── LICENSE ← MIT License

~~~
▶ Quickstart
1️⃣ Run in Colab (no setup required)
1.	Click the Open in Colab badge above.
2.	Runtime → Run all.
3.	Watch each section:
•	• Data load & split
•	• Preprocessing
•	• TF-IDF + training (3 epochs baseline)
•	• Evaluation (report & heatmap)
•	• Save *.joblib artifacts
4.	Done! No local GPU or installs needed.
2️⃣ Local Setup
bash
git clone https://github.com/YourUsername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# create & activate virtual env
python3 -m venv .venv && source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
bash
python src/train.py \
  --data-path path/to/IMDBDataset.csv \
  --model-out sentiment_lr_model.joblib \
  --vectorizer-out tfidf_vectorizer.joblib
bash
python src/predict.py \
  --model sentiment_lr_model.joblib \
  --vectorizer tfidf_vectorizer.joblib \
  --text "That movie was phenomenal! Best I've seen all year."
📈 Sample Results
Stage	Accuracy
LogisticRegression (base)	0.88
LogisticRegression (tuned)	0.90
⚠ Critical Considerations
•	- **Vocabulary Size:** TF-IDF may blow up memory on rare tokens. Consider trimming max_features or using hashing.
•	- **Imbalance:** We use class_weight='balanced', but you might try focal loss or SMOTE for fine-tuning.
•	- **Model Choice:** A simple linear model is fast—swap in an SVM or even BERT for higher accuracy.
•	- **Preprocessing Sanity:** Always sanity-check on random text or blanks to catch tokenization bugs.
🤝 Contributing
1. Fork this repo
2. Create a new branch (git checkout -b feat/my-feature)
3. Commit your changes (git commit -m "Add awesome feature")
4. Push (git push origin feat/my-feature)
5. Open a Pull Request

