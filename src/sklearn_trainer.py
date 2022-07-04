import pickle
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import wandb
import xgboost as xgb
from absl import app, flags, logging
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import make_scorer, hamming_loss
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.dummy import DummyClassifier

from src.sklearn_utils import construct_pipeline, get_naive_baselines

flags.DEFINE_string("data_dir", "data", "Data directory")
flags.DEFINE_string("wandb_project", "wandb_project", "Wandb project name")
flags.DEFINE_integer("most_common_labels", 10, "Number of most common labels")
flags.DEFINE_string("model", "gradient_boosting", "Model to use")
flags.DEFINE_float("test_set_frac", 0.1, "Fraction of data to use for test set")
flags.DEFINE_integer("cv_folds", 5, "Number of folds for cross-validation")
flags.DEFINE_list("max_depth", [2, 4, 6], "Max depth of trees")  # 2, 4, 6

FLAGS = flags.FLAGS


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data.

    This really belongs as part of the preprocessing pipeline, but it's easier to do it here for now
    as we don't need to deploy this.

    Args:
        df: Features and labels.

    Returns:
        pd.DataFrame with preprocessed data.

    """
    df["full_text"] = df["policy_title"] + ": " + df["description_text"]
    df["sectors_list"] = df["sectors"].str.split(";")
    df = df[~df.description_text.duplicated()]
    # Only keep sectors in top n labels
    common_labels = (
        df["sectors_list"].explode().value_counts().index[0 : FLAGS.most_common_labels]
    )
    df["sectors_list"] = df["sectors_list"].apply(
        lambda x: [tag for tag in x if tag in common_labels]
    )
    df_filtered = df.copy()[df["sectors_list"].str.len() > 0]
    df_filtered.drop(columns="sectors", inplace=True)
    return df_filtered


def main(_):
    clf = xgb.XGBClassifier(tree_method="gpu_hist", single_precision_histogram=True)
    # Show all messages, including ones pertaining to debugging
    xgb.set_config(verbosity=2)
    wandb.init(project="climate-multi-label-classification", entity="stefl14")
    logging.info("Loading data")
    data_dir = Path(__file__).parent.parent / "data" / "recruitment-task_1-full.csv"
    df = pd.read_csv(data_dir)
    df_filtered = preprocess_data(df)

    # One-hot encode labels.
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df_filtered["sectors_list"])

    logging.info("Splitting data")
    # Use a smarter stratified split than the default. Tries to preserve the proportion of each label in each fold.
    mlsd = MultilabelStratifiedKFold(
        n_splits=FLAGS.cv_folds, shuffle=True, random_state=42
    )
    for train_index, test_index in mlsd.split(df_filtered, y):
        df_train, df_test = df_filtered.iloc[train_index], df_filtered.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        break

    X, y = df_train, y_train

    full_pipe_gb = construct_pipeline(
        clf=clf,
    )

    param_grid_gb = {
        "clf__base_estimator__max_depth": FLAGS.max_depth,
    }

    logging.info("Fitting naive baselines")
    # naive_baselines = get_naive_baselines(X, y, df_test)
    naive_baselines = {}
    for strategy in ("most_frequent", "stratified", "prior"):
        dd = DummyClassifier(strategy=strategy)
        dd.fit(X, y)
        y_naive = dd.predict(df_test)
        naive_baselines[strategy] = dd
        # hl = hamming_loss(y_true, y_naive)
        # print(
        #     f"Naive baseline to beat on Hamming Loss for strategy {strategy} is: {hl}"
        # )
    grid = GridSearchCV(
        full_pipe_gb,
        param_grid=param_grid_gb,
        scoring=make_scorer(hamming_loss),
        cv=MultilabelStratifiedKFold(
            n_splits=FLAGS.cv_folds, shuffle=True, random_state=42
        ),
        error_score="raise",
    )
    logging.info("Fitting models")
    grid.fit(X, y)
    # mean score in each trial
    logging.info(f'Mean score: {grid.cv_results_["mean_test_score"].mean()}')

    # Pickle results
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    model_name = f"{FLAGS.model}-{timestamp}"
    # dump the grid search.
    joblib.dump(grid, artifacts_dir / f"{model_name}.joblib")
    # Save the naive baselines to disk so we can use them later.
    for k, v in naive_baselines.items():
        joblib.dump(v, artifacts_dir / f"naive-{k}.joblib")
    # write df_test to csv
    df_test.to_csv(artifacts_dir / f"{model_name}-test.csv", index=False)

    # pickle the binarizer
    with open(artifacts_dir / f"{model_name}-binarizer.pkl", "wb") as f:
        pickle.dump(mlb, f)

    results_table = pd.DataFrame(grid.cv_results_)

    table = wandb.Table(data=results_table, columns=results_table.columns)
    wandb.log({"table": table})


if __name__ == "__main__":
    app.run(main)
