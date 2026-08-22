"""Text cleaning and NLTK-based preprocessing shared by training and the app."""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_REQUIRED_NLTK_DATA = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

for path, package in _REQUIRED_NLTK_DATA:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_HTML_TAG_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/punctuation/digits, tokenize, remove stopwords, lemmatize."""
    text = str(text).lower()
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _NON_ALPHA_RE.sub(" ", text)

    tokens = word_tokenize(text)
    tokens = [
        _LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in _STOPWORDS and len(token) > 2
    ]
    return " ".join(tokens)


def score_to_sentiment(score: int) -> str:
    """Map the dataset's 1-5 Score to a 3-class sentiment label."""
    if score >= 4:
        return "Positive"
    if score == 3:
        return "Neutral"
    return "Negative"
