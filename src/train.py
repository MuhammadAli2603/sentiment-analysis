import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import preprocess_text

def main(args):
    # 1. Load & preprocess
    df = pd.read_csv(args.data_path)
    df['clean'] = df['review'].apply(preprocess_text)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )

    # 3. TF-IDF
    tfidf = TfidfVectorizer(max_features=20_000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    # 4. Train
    lr = LogisticRegression(class_weight='balanced', max_iter=1_000)
    lr.fit(X_train_tfidf, y_train)

    # 5. Evaluate
    preds = lr.predict(X_test_tfidf)
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds, labels=['positive','negative'])
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=['pos','neg'], yticklabels=['pos','neg'])
    plt.title("Confusion Matrix")
    plt.show()

    # 6. Save artifacts
    joblib.dump(tfidf, args.vectorizer_out)
    joblib.dump(lr, args.model_out)
    print(f"Saved vectorizer → {args.vectorizer_out}")
    print(f"Saved model      → {args.model_out}")

if _name_ == "_main_":
    p = argparse.ArgumentParser()
    p.add_argument('--data-path',       type=str, required=True)
    p.add_argument('--vectorizer-out',  type=str, default='tfidf_vectorizer.joblib')
    p.add_argument('--model-out',       type=str, default='sentiment_lr_model.joblib')
    args = p.parse_args()
    main(args)