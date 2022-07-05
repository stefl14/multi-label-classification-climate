import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn.pipeline
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import hamming_loss
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer


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


def train_test_split_multilabel(
    df: pd.DataFrame,
    y,
    num_folds: int = 5,
) -> None:
    """Split data into train and test sets using a smart way to split multilabel data.

    Returns:
        pd.DataFrame, pd.DataFrame, np.array, np.array

    """
    mlsd = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    for train_index, test_index in mlsd.split(df, y):
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        break
    return df_train, df_test, y_train, y_test


def select_col(df, col="full_text"):
    return df[
        col
    ]  # lambda's not picklable in sklearn pipeline, hence this strange looking one-liner.


def construct_pipeline(
    clf: BaseEstimator = RandomForestClassifier(), max_df: float = 0.5
) -> sklearn.pipeline.Pipeline:
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


def predict_fn(df: pd.DataFrame, mlb: MultiLabelBinarizer, model):
    y_true = mlb.transform(df["sectors_list"])
    y_pred = model.predict(df)
    hl = hamming_loss(y_true, y_pred)
    return y_true, y_pred, hl


def plot_one_vs_rest_success_rates(y_true, y_pred, y_pred_naive, classes):
    """Plot one vs rest success rates for a naive baseline and a classifier.
    Args:
        y_true: actual labels
        y_pred: predicted labels by classifier
        y_pred_naive: predicted labels by naive baseline

    Returns:
        go.Figure

    """
    success_df = pd.DataFrame()
    success_df["xgb_success"] = 100 * (y_true == y_pred).sum(axis=0) / len(y_true)
    success_df["naive_success"] = (
        100 * (y_true == y_pred_naive).sum(axis=0) / len(y_true)
    )
    # plot success rate for each class using column names as labels
    fig = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=success_df["xgb_success"],
                name="OneVsRest Success Rate XGB",
            ),
            go.Bar(
                x=classes,
                y=success_df["naive_success"],
                name="OneVsRest Success Rate Naive",
            ),
        ]
    )
    fig.update_layout(barmode="group")
    # update title
    fig.update_layout(title_text="Success Rate for Each Class for Naive and XGB")
    return fig


def plot_label_dist(
    y_true, mlb: MultiLabelBinarizer
) -> None:
    """Plot distribution of labels for each class.

    Args:
        y_true:
        mlb:

    Returns:
        go.Figure

    """
    true_test_distribution = y_true.sum(axis=0)

    fig = go.Figure(
        data=[
            go.Bar(
                name="ground truth distribution",
                x=mlb.classes_,
                y=true_test_distribution,
                yaxis="y",
                offsetgroup=1,
            ),
        ],
        layout={
            "yaxis": {"title": "SF Zoo axis"},
        },
    )

    # Change the bar mode
    fig.update_layout(
        barmode="group", title="Predicted vs True distribution on unseen test set."
    )
    return fig


def get_naive_baselines(
    X: np.array, y: np.array, y_true: np.array, df_test: pd.DataFrame
) -> None:
    """
    Get naive baselines for the test set.

    Args:
        X:
        y:
        y_true:
        df_test:

    Returns:

    """
    naive_baselines = {}
    for strategy in ("most_frequent", "stratified", "prior"):
        dd = DummyClassifier(strategy=strategy)
        dd.fit(X, y)
        y_naive = dd.predict(df_test)
        naive_baselines[strategy] = dd
        hl = hamming_loss(y_true, y_naive)
        print(
            f"Naive baseline to beat on Hamming Loss for strategy {strategy} is: {hl}"
        )
    return naive_baselines


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
