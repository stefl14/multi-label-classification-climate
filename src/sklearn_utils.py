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
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations. Function needed
    as library doesn't give option for just one split.'
    """
    stratifier = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[1.0-train_size, train_size, ])
    train_indices, test_indices = next(stratifier.split(X, y))
    if type(X)==pd.DataFrame:
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    else:
        X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

def simple_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply simple feature engineering to text columns. This is by no means
    an exhaustive list of what we could do, just a simple initial experiment that
    can be added to. For example, we could add proportion of stop words, number of
    html tags, number of capital letters etc. There is a large number of possibilities
    for what could be added to this pipeline, informed by more in-depth EDA.
    :param df: Input df, contains raw text.
    :return:
    """
    # TODO: Add to this feature engineering pipeline after more in depth eda.
    df=df.copy()
    df['title_len'] = df['policy_title'].str.len()
    df['description_len']=df['description_text'].str.len()
    df['num_sentences_description']=df['description_text'].str.split('.').str.len()
    df['num_sentences_title']=df['description_text'].str.split('.').str.len()
    df['num_words_description']=df['description_text'].str.split().str.len()
    df['num_words_title']=df['policy_title'].str.split().str.len()
    return df

def select_col(df, col='full_text'):
    return df[col] #lambda's not picklable in sklearn pipeline, hence this strange looking one-liner.

from sklearn.metrics import hamming_loss

def construct_pipeline(max_df: float = 0.5) -> sklearn.pipeline.Pipeline:
    """
    An sklearn pipeline that concatenates text and engineered features.
    I've hardcoded some variables here rather than parameterising them because I haven't
    done much formal experimentation. But obviously in a non rushed section this function
    would be parameterised with all the hyperparams.

    This function returns something that can be binarised into pickle format (assuming python
    run time) for easy deployment. This function carries out all preprocessing within the pipeline
    so that there is no train-serving skew, and so it's easy to tune any step of the pipeline
    as a hyperparameter.

    This pipeline does assume the text request is formatted into a dataframe, even if it's not
    a batch request (say if someone made a query for a concept). This is a weakness and an
    artifact of some custom processing. It could easily be fixed up for deployment.

    :param max_df: Set a lower value if you want more corpus specific stop words (these generally add noise).
    :return:
    """
    pipeline_text = Pipeline([
    ("select_text_col",FunctionTransformer(select_col)), # I know there's a more canonical way of doing this line, but I reached a bug so hacking for speed.
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=False, ngram_range=(1,1),min_df=0.2,max_df=max_df)),
    ])


    full_pipe = Pipeline([
        ('features', FeatureUnion([
            ('pipeline_text', pipeline_text),
            ('feature_engineering',TextFeatureEngineering()),
            ])),
         ('clf', ClassifierChain(RandomForestClassifier()))
    ]
    )
    return full_pipe

class TextFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Transformer to pre-process non-text features to fit into a deployable sklearn pipeline.
    This transformer is far from an exhaustive when it comes to representing non-text features.
    We could include everything from number of capital letters, fraction of stop words, we could
    use spacy to get named entities etc. Note the use of logs to reflect the power law-esque
    distributions of text lengths.
    """
    def __init__(self, apply_log_transform=False):
        self.apply_log_transform = apply_log_transform
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X=X.copy()
        X['full_text'] = X['policy_title']+': '+X['description_text']
        if self.apply_log_transform:
            X['num_sentences_description']=np.log(X['description_text'].str.split('.').str.len().astype(float))
            X['num_sentences_title']=np.log(X['description_text'].str.split('.').str.len().astype(float))
            X['num_words_description']=np.log(X['description_text'].str.split().str.len().astype(float))
            X['num_words_title']=np.log(X['policy_title'].str.split().str.len().astype(float))
        else:
            X['num_sentences_description']=X['description_text'].str.split('.').str.len()
            X['num_sentences_title']=X['description_text'].str.split('.').str.len()
            X['num_words_description']=X['description_text'].str.split().str.len()
            X['num_words_title']=X['policy_title'].str.split().str.len()
        X=X[['num_sentences_description','num_sentences_title','num_words_description','num_words_title']]
        return X.to_numpy()