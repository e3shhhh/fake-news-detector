"""Smoke tests using a tiny synthetic dataset (no downloads needed)."""

import pandas as pd

from fakenews.predict import predict
from fakenews.preprocess import FAKE, REAL, clean_text, strip_agency_tag
from fakenews.train import train


def test_strip_agency_tag():
    assert strip_agency_tag("WASHINGTON (Reuters) - Senate passes bill").startswith("Senate")
    assert strip_agency_tag("No tag here") == "No tag here"


def test_clean_text_removes_urls_and_stopwords():
    out = clean_text("Check https://example.com the BREAKING news!!!", stem=False)
    assert "http" not in out and "the" not in out.split()
    assert "breaking" in out and "news" in out


def _synthetic() -> pd.DataFrame:
    fake = [f"shocking secret miracle cure exposed conspiracy hoax number {i}" for i in range(60)]
    real = [f"government committee reports quarterly budget figures statement {i}" for i in range(60)]
    return pd.DataFrame(
        {"text": fake + real, "label": [FAKE] * len(fake) + [REAL] * len(real)}
    )


def test_train_and_predict():
    model, metrics = train(_synthetic(), min_df=1)
    assert metrics["accuracy"] > 0.9

    fake = predict(model, "shocking secret miracle cure exposed conspiracy hoax")
    real = predict(model, "government committee reports quarterly budget statement")
    assert fake.label == "Fake" and real.label == "Real"
    assert 0 <= fake.credibility <= 100 and fake.credibility < real.credibility
