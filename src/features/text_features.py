"""
NLP feature engineering for the Email Security AI project.

This module creates TF-IDF representations from email text.

Two representations are created:

1. Word-level TF-IDF
2. Character-level TF-IDF

Both are useful for email security because attackers can
change wording while preserving suspicious patterns.
"""

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TextFeaturePipeline:
    """
    Stores the fitted TF-IDF vectorizers.

    The vectorizers must be fitted only on training data.
    """

    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer


def create_word_vectorizer() -> TfidfVectorizer:
    """
    Create the word-level TF-IDF vectorizer.

    ngram_range=(1, 2) means:
        unigram:  "verify"
        bigram:   "verify account"
    """

    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=100_000,
    )


def create_char_vectorizer() -> TfidfVectorizer:
    """
    Create the character-level TF-IDF vectorizer.

    Character n-grams can capture patterns that word-level
    tokenization may miss.
    """

    return TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )


def fit_text_pipeline(
    training_text: pd.Series,
) -> TextFeaturePipeline:
    """
    Fit both TF-IDF vectorizers using training text only.

    This is important for preventing data leakage.
    """

    word_vectorizer = create_word_vectorizer()

    char_vectorizer = create_char_vectorizer()

    word_vectorizer.fit(training_text)

    char_vectorizer.fit(training_text)

    return TextFeaturePipeline(
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
    )


def transform_text(
    pipeline: TextFeaturePipeline,
    text: pd.Series,
):
    """
    Transform text using an already-fitted pipeline.

    Returns a sparse matrix containing:
        word TF-IDF + character TF-IDF
    """

    word_features = (
        pipeline.word_vectorizer.transform(text)
    )

    char_features = (
        pipeline.char_vectorizer.transform(text)
    )

    return hstack(
        [
            word_features,
            char_features,
        ],
        format="csr",
    )


def print_text_feature_summary(
    pipeline: TextFeaturePipeline,
    feature_matrix,
) -> None:
    """
    Print information about the generated NLP feature matrix.
    """

    word_count = len(
        pipeline.word_vectorizer.get_feature_names_out()
    )

    char_count = len(
        pipeline.char_vectorizer.get_feature_names_out()
    )

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — NLP FEATURE SUMMARY")
    print("=" * 70)

    print(
        f"\nWord TF-IDF features: "
        f"{word_count:,}"
    )

    print(
        f"Character TF-IDF features: "
        f"{char_count:,}"
    )

    print(
        f"Combined features: "
        f"{feature_matrix.shape[1]:,}"
    )

    print(
        f"Email records: "
        f"{feature_matrix.shape[0]:,}"
    )

    print(
        f"Matrix type: "
        f"{type(feature_matrix).__name__}"
    )

    print(
        f"Non-zero values: "
        f"{feature_matrix.nnz:,}"
    )


if __name__ == "__main__":

    from src.data.spamassassin_loader import (
        load_spamassassin,
    )

    from src.preprocessing.cleaner import (
        clean_dataset,
    )

    dataframe = load_spamassassin()

    dataframe, _ = clean_dataset(
        dataframe
    )

    text = dataframe["email_text"]

    pipeline = fit_text_pipeline(
        text
    )

    feature_matrix = transform_text(
        pipeline,
        text,
    )

    print_text_feature_summary(
        pipeline,
        feature_matrix,
    )
