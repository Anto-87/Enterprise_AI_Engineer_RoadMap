"""Streamlit UI: type a food review, get the predicted sentiment."""
import pickle
from pathlib import Path

import streamlit as st

from preprocess import clean_text

BASE_DIR = Path(__file__).resolve().parent
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"

SENTIMENT_STYLE = {
    "Positive": "green",
    "Neutral": "orange",
    "Negative": "red",
}

SAMPLE_REVIEWS = {
    "Positive example": (
        "This coffee is fantastic! Rich flavor, smooth finish, and it arrived "
        "well before the estimated delivery date. Will definitely order again."
    ),
    "Neutral example": (
        "It's an okay snack. Not bad, not great — tastes about the same as the "
        "store brand but costs a bit more."
    ),
    "Negative example": (
        "Very disappointed. The jar arrived broken and the product inside had "
        "clearly gone stale. Would not recommend."
    ),
}


@st.cache_resource
def load_artifacts():
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return vectorizer, model


def predict_sentiment(text, vectorizer, model):
    cleaned = clean_text(text)
    if not cleaned:
        return None
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(vec)[0]))
    return prediction, probabilities


def render_result(review_text, vectorizer, model):
    if not review_text.strip():
        st.warning("Please enter some review text first.")
        return

    result = predict_sentiment(review_text, vectorizer, model)
    if result is None:
        st.warning("Couldn't extract meaningful words from that text. Try a longer review.")
        return

    prediction, probabilities = result
    color = SENTIMENT_STYLE.get(prediction, "blue")

    st.markdown(f"#### Predicted sentiment: :{color}[{prediction}]")
    ordered = sorted(probabilities.items(), key=lambda x: -x[1])
    bar_cols = st.columns(len(ordered))
    for col, (cls, prob) in zip(bar_cols, ordered):
        col.progress(float(prob), text=f"{cls}: {prob:.0%}")


st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="centered",
)

st.markdown(
    "<style>div.block-container{padding-top:2rem;padding-bottom:1rem;}</style>",
    unsafe_allow_html=True,
)

st.title("Sentiment Analyzer")

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    with st.spinner(
        "No trained model found — training one now on the Fine Food Reviews "
        "dataset. This takes a few minutes and only happens once."
    ):
        import train as train_module

        train_module.main()
    st.rerun()

vectorizer, model = load_artifacts()

with st.container(border=True):
    analyzer_tab, sample_tab, about_tab = st.tabs(
        ["Analyzer", "Try a Sample", "About the Model"]
    )

    with analyzer_tab:
        custom_text = st.text_area(
            "Review text",
            height=100,
            placeholder="e.g. This coffee tastes amazing and arrived quickly!",
            key="custom_review_text",
            label_visibility="collapsed",
        )
        if st.button(
            "Analyze Sentiment", type="primary", use_container_width=True, key="analyze_custom"
        ):
            render_result(custom_text, vectorizer, model)

    with sample_tab:
        sample_label = st.radio(
            "Sample review",
            options=list(SAMPLE_REVIEWS.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
        st.text_area(
            "Preview",
            value=SAMPLE_REVIEWS[sample_label],
            height=100,
            disabled=True,
            label_visibility="collapsed",
        )
        if st.button(
            "Analyze Sentiment", type="primary", use_container_width=True, key="analyze_sample"
        ):
            render_result(SAMPLE_REVIEWS[sample_label], vectorizer, model)

    with about_tab:
        st.markdown(
            "Sentiment classifier trained on the **Fine Food Reviews** dataset "
            "(500k+ customer reviews) — **NLTK** cleaning + **TF-IDF** + "
            "**Logistic Regression**. Labels: Score 4-5 = Positive, 3 = Neutral, "
            "1-2 = Negative."
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "83%")
        col2.metric("Positive F1", "0.91")
        col3.metric("Negative F1", "0.73")
        col4.metric("Neutral F1", "0.43")
