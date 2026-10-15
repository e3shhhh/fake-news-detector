"""Load a trained model and score articles.

Usage:
    python -m fakenews.predict "Paste an article here..."
    python -m fakenews.predict --file article.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib

from .preprocess import FAKE, REAL, LABEL_NAMES

DEFAULT_MODEL = Path("models/fake_news_model.joblib")


@dataclass
class Prediction:
    label: str
    credibility: float  # 0-100, probability the article is real
    confidence: float  # 0-100, probability of the predicted label


def load_model(path: str | Path = DEFAULT_MODEL):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run `python -m fakenews.train` first.")
    return joblib.load(path)


def predict(model, text: str) -> Prediction:
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    p_real, p_fake = proba[classes.index(REAL)], proba[classes.index(FAKE)]
    label = FAKE if p_fake > p_real else REAL
    return Prediction(
        label=LABEL_NAMES[label],
        credibility=round(float(p_real) * 100, 1),
        confidence=round(float(max(p_real, p_fake)) * 100, 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Article text")
    parser.add_argument("--file", help="Read the article from a text file")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    text = Path(args.file).read_text() if args.file else args.text
    if not text:
        parser.error("Provide article text or --file.")

    result = predict(load_model(args.model), text)
    print(f"Verdict     : {result.label}")
    print(f"Credibility : {result.credibility}/100")
    print(f"Confidence  : {result.confidence}%")


if __name__ == "__main__":
    main()
