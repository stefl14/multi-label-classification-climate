import pandas as pd
import numpy as np
import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.model_selection import IterativeStratification
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import ClassifierChain


def iter_train_test_split(X, y, train_size: float):
    """
    Custom iterative stratification for train-test split (skmultilearn doesn't support a single split).

    Args:
        X: features
        y: label
        train_size:

    Returns:

    """
    stratifier = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[
            1.0 - train_size,
            train_size,
        ],
    )
    train_indices, test_indices = next(stratifier.split(X, y))
    if type(X) == pd.DataFrame:
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    else:
        X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


def simple_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    A simple feature engineering function that adds some engineered features.

    Args:
        df: pd.DataFrame of text features.

    Returns:
        pd.DataFrame with additional calculated features.

    """
    # TODO: Add to this after more in depth eda.
    df = df.copy()
    df["title_len"] = df["policy_title"].str.len()
    df["description_len"] = df["description_text"].str.len()
    df["num_sentences_description"] = df["description_text"].str.split(".").str.len()
    df["num_sentences_title"] = df["description_text"].str.split(".").str.len()
    df["num_words_description"] = df["description_text"].str.split().str.len()
    df["num_words_title"] = df["policy_title"].str.split().str.len()
    return df


def select_col(df, col="full_text"):
    return df[
        col
    ]  # lambda's not picklable in sklearn pipeline, hence this strange looking one-liner.



def construct_pipeline(clf: BaseEstimator = RandomForestClassifier(), max_df: float = 0.5) -> sklearn.pipeline.Pipeline:
    """
    Concatenate text and engineered features.

    Carries out all preprocessing within pipeline to avoid train-serve skew and
    returns something that can be put into a pickle binary (useful for Python runtime).

    Assumes request formatted as a dataframe, something we can fix up later.

    Args: max_df: float, maximum document frequency for TF-IDF. To sparsify feature set by ignore likely
    uninformative words.

    clf: BaseEstimator, sklearn classifier to use.

    Returns:
        sklearn.pipeline.Pipeline

    """
    pipeline_text = Pipeline(
        [
            (
                "select_text_col",
                FunctionTransformer(select_col),
            ),  # There's a more canonical way of doing this line, but I reached a bug so hacking for speed.
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    lowercase=False,
                    ngram_range=(1, 1),
                    max_df=max_df,
                ),
            ),
        ]
    )

    full_pipe = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("pipeline_text", pipeline_text),
                        ("feature_engineering", TextFeatureEngineering()),
                    ]
                ),
            ),
            ("clf", ClassifierChain(clf)),
        ]
    )
    return full_pipe


class TextFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Transformer to produce features from text to put into a deployable sklearn pipeline.

    e.g. log transforming text length to reflect the power law-esque
    distributions found in EDA.

    The features here are not exhaustive but represent a start.
    """

    def __init__(self, apply_log_transform=False):
        self.apply_log_transform = apply_log_transform

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["full_text"] = X["policy_title"] + ": " + X["description_text"]
        if self.apply_log_transform:
            X["num_sentences_description"] = np.log(
                X["description_text"].str.split(".").str.len().astype(float)
            )
            X["num_sentences_title"] = np.log(
                X["description_text"].str.split(".").str.len().astype(float)
            )
            X["num_words_description"] = np.log(
                X["description_text"].str.split().str.len().astype(float)
            )
            X["num_words_title"] = np.log(
                X["policy_title"].str.split().str.len().astype(float)
            )
        else:
            X["num_sentences_description"] = (
                X["description_text"].str.split(".").str.len()
            )
            X["num_sentences_title"] = X["description_text"].str.split(".").str.len()
            X["num_words_description"] = X["description_text"].str.split().str.len()
            X["num_words_title"] = X["policy_title"].str.split().str.len()
        X = X[
            [
                "num_sentences_description",
                "num_sentences_title",
                "num_words_description",
                "num_words_title",
            ]
        ]
        return X.to_numpy()
