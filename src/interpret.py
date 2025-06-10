import argparse
import joblib
from src.preprocess import preprocess_text

def predict(model_path: str, vect_path: str, text: str):
    vect = joblib.load(vect_path)
    model = joblib.load(model_path)
    clean = preprocess_text(text)
    tfidf_vec = vect.transform([clean])
    proba = model.predict_proba(tfidf_vec)[0][1]
    label = 'positive' if proba > 0.5 else 'negative'
    print(f"Review: {text!r}")
    print(f"→ Predicted: {label} (p={proba:.3f})")

if _name_ == "_main_":
    p = argparse.ArgumentParser()
    p.add_argument('--model',      type=str, required=True)
    p.add_argument('--vectorizer', type=str, required=True)
    p.add_argument('--text',       type=str, required=True)
    args = p.parse_args()
    predict(args.model, args.vectorizer, args.text)