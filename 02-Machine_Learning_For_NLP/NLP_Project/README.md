# Fine Food Reviews — Sentiment Analysis

NLTK-based text preprocessing + TF-IDF + Logistic Regression classifier for the
[Fine Food Reviews](../dataset/Reviews.csv) dataset, with a Streamlit UI.

Sentiment labels are derived from the `Score` column:

- Score 4-5 → **Positive**
- Score 3 → **Neutral**
- Score 1-2 → **Negative**

## Files

- `preprocess.py` — shared text cleaning (lowercase, strip HTML/punctuation, tokenize, remove stopwords, lemmatize via NLTK)
- `train.py` — loads `Reviews.csv`, cleans text, trains TF-IDF + Logistic Regression, evaluates, saves the model
- `app.py` — Streamlit app: enter a review, get the predicted sentiment + confidence
- `models/` — saved `tfidf_vectorizer.pkl` and `sentiment_model.pkl` (created by `train.py`)

## Usage

```bash
# 1. Train the model (trains on a stratified 150k-row sample — a few minutes)
python train.py

# 2. Launch the UI
python -m streamlit run app.py
```

The app also auto-trains the model on first launch if `models/` is empty.
