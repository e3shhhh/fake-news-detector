"""Text cleaning and dataset loading."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Label convention used everywhere in this project.
REAL, FAKE = 0, 1
LABEL_NAMES = {REAL: "Real", FAKE: "Fake"}

_stemmer = PorterStemmer()

_URL = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+")
_NON_LETTERS = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")

# The ISOT / Kaggle "Fake and Real News" dataset has a well-known leak: most
# real articles begin with a location and "(Reuters) -". A model can reach
# very high accuracy just by spotting that tag, so we strip it.
_AGENCY_TAG = re.compile(r"^.{0,80}?\(\s*reuters\s*\)\s*[-–—]*\s*", re.IGNORECASE)


def strip_agency_tag(text: str) -> str:
    """Remove a leading 'CITY (Reuters) -' style tag from an article."""
    return _AGENCY_TAG.sub("", text, count=1)


def clean_text(text: str, stem: bool = True) -> str:
    """Lowercase, drop URLs/punctuation/stop-words, and optionally stem."""
    if not isinstance(text, str):
        return ""
    text = strip_agency_tag(text)
    text = _URL.sub(" ", text.lower())
    text = _NON_LETTERS.sub(" ", text)
    tokens = [t for t in _SPACES.split(text) if len(t) > 2 and t not in ENGLISH_STOP_WORDS]
    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


def load_dataset(data_dir: str | Path) -> pd.DataFrame:
    """Load Fake.csv and True.csv from ``data_dir`` into one labelled frame.

    Returns columns: ``text`` (title + body) and ``label`` (0 = real, 1 = fake).
    """
    data_dir = Path(data_dir)
    fake_path, true_path = data_dir / "Fake.csv", data_dir / "True.csv"
    for p in (fake_path, true_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Download the dataset first - see data/README.md."
            )

    fake = pd.read_csv(fake_path).assign(label=FAKE)
    real = pd.read_csv(true_path).assign(label=REAL)
    df = pd.concat([fake, real], ignore_index=True)

    df["text"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.strip()
    df = df[["text", "label"]]
    df = df[df["text"].str.len() > 20].drop_duplicates(subset="text")
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)
