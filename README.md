# 🔍 Fake Review Detection System

A hybrid **Genetic Algorithm + Hidden Markov Model** system for detecting fraudulent Amazon product reviews.

## Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐
│   Scraper    │───▶│  Preprocessing   │───▶│ Feature Extraction│
│  (Amazon)    │    │  (NLP Pipeline)  │    │ (TF-IDF + More)   │
└──────────────┘    └──────────────────┘    └─────────┬─────────┘
                                                       │
                    ┌──────────────────┐               │
                    │   HMM Engine     │───────────────┤
                    │ (Behavior Model) │               │
                    └──────────────────┘               ▼
                                            ┌───────────────────┐
                    ┌──────────────────┐    │  Feature Fusion   │
                    │    GA Engine     │◀───┤  (Combined Matrix)│
                    │(Feature Select.) │    └───────────────────┘
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐    ┌───────────────────┐
                    │ Hybrid Classifier│───▶│    Dashboard      │
                    │ (RF+GBM+LR)     │    │   (Streamlit)     │
                    └──────────────────┘    └───────────────────┘
```

## Features

- **Web Scraping**: Automated Amazon review collection with anti-detection
- **NLP Pipeline**: Text cleaning, tokenization, sentiment analysis, linguistic features
- **Feature Engineering**: TF-IDF, behavioral, temporal, and metadata features
- **Genetic Algorithm**: Parallel feature selection with DEAP
- **Hidden Markov Model**: Sequential behavior pattern detection
- **Hybrid Classifier**: Ensemble (Random Forest + Gradient Boosting + Logistic Regression)
- **Interactive Dashboard**: Streamlit-based visualization with Plotly charts

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (with synthetic data)
python app.py --synthetic --samples 1000

# Launch dashboard
python app.py --dashboard

# Run with your own data
python app.py --data data/your_reviews.csv
```

## Project Structure

```
fake_review_detection_project/
├── app.py                          # Main pipeline orchestrator
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── data/                           # Datasets
│   ├── amazon_reviews.csv
│   └── scraped_reviews.csv
├── scraper/                        # Web scraping module
│   └── amazon_scraper.py
├── preprocessing/                  # Text preprocessing
│   └── text_preprocessing.py
├── features/                       # Feature engineering
│   └── feature_extraction.py
├── ga_engine/                      # Genetic algorithm
│   └── parallel_genetic_optimizer.py
├── hmm_engine/                     # Hidden Markov Model
│   └── hidden_markov_analyzer.py
├── hybrid_model/                   # Ensemble classifier
│   └── hybrid_classifier.py
├── dashboard/                      # Streamlit dashboard
│   └── visual_dashboard.py
└── utils/                          # Utility functions
    └── helpers.py
```

## Pipeline Steps

1. **Data Loading** — Load reviews from CSV or generate synthetic data
2. **Text Preprocessing** — Clean, tokenize, extract sentiment & linguistic features
3. **Feature Extraction** — Build TF-IDF + behavioral + temporal + metadata features
4. **HMM Analysis** — Fit Hidden Markov Model to detect behavioral anomalies
5. **GA Optimization** — Evolve optimal feature subsets using genetic algorithm
6. **Classification** — Train hybrid ensemble on GA-selected + HMM features
7. **Results** — Save metrics, model, and launch dashboard

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| NLP | NLTK, TextBlob, scikit-learn |
| ML | scikit-learn, hmmlearn |
| GA | DEAP (Distributed Evolutionary Algorithms) |
| Dashboard | Streamlit, Plotly |
| Scraping | BeautifulSoup, Requests |
