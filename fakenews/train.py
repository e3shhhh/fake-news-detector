"""Train the detector and compare it with a baseline.

Usage:
    python -m fakenews.train --data-dir data --out-dir models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .preprocess import LABEL_NAMES, clean_text, load_dataset


def build_model(min_df: int = 5) -> Pipeline:
    """TF-IDF (1-2 grams) + Logistic Regression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    max_features=50_000,
                    min_df=min_df,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, C=5.0)),
        ]
    )


def build_baseline(min_df: int = 5) -> Pipeline:
    """Simple bag-of-words + Naive Bayes baseline for comparison."""
    return Pipeline(
        [
            ("bow", CountVectorizer(preprocessor=clean_text, min_df=min_df)),
            ("clf", MultinomialNB()),
        ]
    )


def train(df: pd.DataFrame, test_size: float = 0.2, min_df: int = 5, seed: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size, stratify=df["label"], random_state=seed
    )

    baseline = build_baseline(min_df).fit(X_train, y_train)
    model = build_model(min_df).fit(X_train, y_train)

    base_pred, pred = baseline.predict(X_test), model.predict(X_test)
    base_err, err = 1 - accuracy_score(y_test, base_pred), 1 - accuracy_score(y_test, pred)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "baseline_accuracy": float(accuracy_score(y_test, base_pred)),
        "error_reduction_vs_baseline": float((base_err - err) / base_err) if base_err else 0.0,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(
            y_test, pred, target_names=[LABEL_NAMES[0], LABEL_NAMES[1]], output_dict=True
        ),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-df", type=int, default=5)
    args = parser.parse_args()

    df = load_dataset(args.data_dir)
    print(f"Loaded {len(df):,} articles ({df['label'].mean():.1%} fake).")

    model, metrics = train(df, args.test_size, args.min_df)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "fake_news_model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Test accuracy      : {metrics['accuracy']:.4f}")
    print(f"Baseline accuracy  : {metrics['baseline_accuracy']:.4f}")
    print(f"Error reduction    : {metrics['error_reduction_vs_baseline']:.1%}")
    print(f"Saved model and metrics to {out}/")


if __name__ == "__main__":
    main()
