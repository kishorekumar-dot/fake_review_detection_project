"""
text_preprocessing.py - NLP Text Preprocessing Pipeline.

Handles text cleaning, tokenization, stopword removal, lemmatization,
and feature-ready text transformations for review analysis.
"""

import re
import string
from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.helpers import get_logger, timer, validate_dataframe, print_header

logger = get_logger("preprocessing")

# Download required NLTK data
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(resource, quiet=True)


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for review analysis.

    Pipeline stages:
        1. Lowercase conversion
        2. URL and email removal
        3. HTML tag stripping
        4. Special character / punctuation removal
        5. Number normalization
        6. Tokenization
        7. Stopword removal
        8. Lemmatization
        9. Sentiment extraction
    """

    def __init__(self, remove_stopwords=True, lemmatize=True, min_word_length=2):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # Add domain-specific stopwords
        self.stop_words.update(["amazon", "product", "item", "buy", "bought", "order", "ordered"])
        logger.info("TextPreprocessor initialized")

    def clean_text(self, text: str) -> str:
        """Apply all cleaning steps to raw text."""
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)          # URLs
        text = re.sub(r"\S+@\S+", "", text)                    # Emails
        text = re.sub(r"<[^>]+>", "", text)                     # HTML tags
        text = re.sub(r"[^\w\s]", " ", text)                    # Special chars
        text = re.sub(r"\d+", " NUM ", text)                    # Numbers
        text = re.sub(r"\s+", " ", text).strip()                # Whitespace
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize cleaned text into words."""
        tokens = word_tokenize(text)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) >= self.min_word_length]
        return tokens

    def get_sentiment(self, text: str) -> dict:
        """Extract sentiment polarity and subjectivity using TextBlob."""
        blob = TextBlob(text if isinstance(text, str) else "")
        return {
            "polarity": round(blob.sentiment.polarity, 4),
            "subjectivity": round(blob.sentiment.subjectivity, 4),
        }

    def extract_linguistic_features(self, text: str) -> dict:
        """Extract linguistic features from raw text."""
        if not isinstance(text, str) or not text.strip():
            return {"char_count": 0, "word_count": 0, "avg_word_len": 0,
                    "sentence_count": 0, "exclamation_count": 0,
                    "question_count": 0, "caps_ratio": 0, "digit_ratio": 0}

        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        alpha_chars = [c for c in text if c.isalpha()]

        return {
            "char_count": len(text),
            "word_count": len(words),
            "avg_word_len": round(np.mean([len(w) for w in words]), 2) if words else 0,
            "sentence_count": len([s for s in sentences if s.strip()]),
            "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
            "caps_ratio": round(sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1), 4),
            "digit_ratio": round(sum(1 for c in text if c.isdigit()) / max(len(text), 1), 4),
        }

    @timer
    def process_dataframe(self, df: pd.DataFrame, text_column: str = "body") -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to a DataFrame.

        Args:
            df: Input DataFrame with review text.
            text_column: Name of the column containing review text.

        Returns:
            DataFrame with added preprocessed columns.
        """
        print_header("Text Preprocessing Pipeline")
        validate_dataframe(df, [text_column])
        df = df.copy()

        logger.info("Processing %d reviews...", len(df))

        # Clean text
        df["cleaned_text"] = df[text_column].fillna("").apply(self.clean_text)

        # Tokenize
        df["tokens"] = df["cleaned_text"].apply(self.tokenize)
        df["processed_text"] = df["tokens"].apply(lambda t: " ".join(t))

        # Sentiment
        sentiments = df[text_column].fillna("").apply(self.get_sentiment)
        df["sentiment_polarity"] = sentiments.apply(lambda s: s["polarity"])
        df["sentiment_subjectivity"] = sentiments.apply(lambda s: s["subjectivity"])

        # Linguistic features
        ling_features = df[text_column].fillna("").apply(self.extract_linguistic_features)
        ling_df = pd.DataFrame(ling_features.tolist(), index=df.index)
        df = pd.concat([df, ling_df], axis=1)

        logger.info("Preprocessing complete. Added %d new columns.", len(ling_df.columns) + 4)
        return df


if __name__ == "__main__":
    # Demo with sample data
    sample_data = pd.DataFrame({
        "body": [
            "This product is AMAZING!!! Best purchase ever!!",
            "Terrible quality. Broke after 2 days. DO NOT BUY.",
            "It's okay, nothing special. Works as described.",
            "FAKE FAKE FAKE! This is clearly a paid review!!!",
            "",
        ]
    })

    preprocessor = TextPreprocessor()
    result = preprocessor.process_dataframe(sample_data)
    print(result[["body", "cleaned_text", "sentiment_polarity", "word_count"]].to_string())
