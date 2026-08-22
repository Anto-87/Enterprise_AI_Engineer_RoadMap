"""Train a sentiment classifier on the Amazon Fine Food Reviews dataset.

Pipeline: NLTK cleaning -> TF-IDF -> Logistic Regression (3-class: Positive/Neutral/Negative)
Trained on the full dataset. Saves the fitted vectorizer + model to models/.
"""
import pickle
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from preprocess import clean_text, score_to_sentiment

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "dataset" / "Reviews.csv"
MODELS_DIR = BASE_DIR / "models"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"

# NLTK's pure-Python tokenizer/lemmatizer is too slow to run over all ~568k rows
# in a reasonable time, so we train on a stratified sample instead.
SAMPLE_SIZE = 150_000


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Loading dataset from {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH, usecols=["Score", "Text"])
    df = df.dropna(subset=["Score", "Text"])
    print(f"Loaded {len(df):,} reviews.")

    print("Mapping scores to sentiment labels ...")
    df["sentiment"] = df["Score"].apply(score_to_sentiment)
    print(df["sentiment"].value_counts())

    if len(df) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE:,} rows (stratified by sentiment) for training speed ...")
        total = len(df)
        parts = []
        for label, group in df.groupby("sentiment"):
            n = min(len(group), int(SAMPLE_SIZE * len(group) / total))
            parts.append(group.sample(n=n, random_state=42))
        df = pd.concat(parts, ignore_index=True)
        print(df["sentiment"].value_counts())

    print("Cleaning text with NLTK (tokenize, stopword removal, lemmatization) ...")
    start = time.time()
    df["clean_text"] = df["Text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    print(f"Cleaned {len(df):,} reviews in {time.time() - start:.1f}s")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )

    print("Vectorizing text with TF-IDF ...")
    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=5)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression classifier ...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train_vec, y_train)

    print("Evaluating on held-out test set ...")
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    print(f"Saving vectorizer to {VECTORIZER_PATH}")
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saving model to {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Done.")


if __name__ == "__main__":
    main()
