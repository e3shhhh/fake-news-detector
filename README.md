# 📰 Fake News Detector

**An NLP tool that scores how credible a news article looks, in real time.**
Paste an article into the Streamlit app and get a Real/Fake verdict, a 0-100 credibility gauge, and a confidence breakdown.

> GitHub repo description (copy into the "About" box):
> *NLP tool that classifies news articles as real or fake using TF-IDF and Logistic Regression, with a real-time Streamlit app and credibility gauge.*
> Suggested topics: `machine-learning` `nlp` `fake-news-detection` `scikit-learn` `tf-idf` `streamlit` `python`

## Features

- **End-to-end NLP pipeline**: text cleaning, stop-word removal, stemming, TF-IDF (unigrams + bigrams) and Logistic Regression, trained on 40,000+ articles.
- **Baseline comparison**: every training run also fits a Bag-of-Words + Naive Bayes baseline and reports the error reduction.
- **Real-time Streamlit app**: credibility-score gauge and Real vs Fake confidence chart built with Plotly.
- **SQLite history**: each check is logged locally so you can review recent predictions.
- **CLI**: score an article from the terminal.
- **Tests**: a pytest smoke test that runs on synthetic data, so no dataset download is needed.

## Tech stack

Python · Pandas · NumPy · Scikit-learn · NLTK · TF-IDF · Streamlit · Plotly · Joblib · SQLite

## Project structure

```
fake-news-detector/
├── app.py                  # Streamlit app
├── fakenews/
│   ├── preprocess.py       # cleaning + dataset loading
│   ├── train.py            # training, baseline, metrics
│   └── predict.py          # load model + score text (also a CLI)
├── tests/test_pipeline.py
├── data/README.md          # how to get the dataset
├── models/                 # trained model is saved here (git-ignored)
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/e3shhhh/fake-news-detector.git
cd fake-news-detector
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. **Get the data.** Put `Fake.csv` and `True.csv` in `data/` (see [data/README.md](data/README.md)).
2. **Train.**
   ```bash
   python -m fakenews.train --data-dir data --out-dir models
   ```
   This prints test accuracy, baseline accuracy and error reduction, and saves the model plus `metrics.json`.
3. **Launch the app.**
   ```bash
   streamlit run app.py
   ```
4. **Or use the CLI.**
   ```bash
   python -m fakenews.predict "Paste an article here..."
   python -m fakenews.predict --file article.txt
   ```

Run the tests with `pytest`.

## Results

Run `python -m fakenews.train` to generate your own numbers; they are written to `models/metrics.json`.
Add your measured accuracy and baseline comparison here after training:

| Model | Accuracy |
|---|---|
| Bag-of-Words + Naive Bayes (baseline) | _fill in_ |
| TF-IDF + Logistic Regression | _fill in_ |

## Notes on the dataset and its limits

- **Leakage guard.** In the ISOT dataset most real articles start with a tag like `WASHINGTON (Reuters) -`. A model can lean on that tag instead of the writing itself, which inflates accuracy. This project strips the tag before training and prediction.
- **Narrow domain.** The data is mostly US political news from 2016-2017. The model learns style and vocabulary of that period, so it may not generalise to other topics, years or sources.
- **Not a fact-checker.** The score reflects how much an article resembles the training data's real or fake examples. It does not verify claims. Treat it as a decision aid.

## Possible improvements

- Evaluate on a second, out-of-domain dataset to measure generalisation.
- Try transformer embeddings (e.g. a fine-tuned BERT) against the TF-IDF model.
- Add SHAP/LIME explanations for the top words behind each verdict.

## License

MIT, see [LICENSE](LICENSE).

## Author

**Eesh Pandya** · [LinkedIn](https://linkedin.com/in/eesh-pandya-83302a256) · [GitHub](https://github.com/e3shhhh)
