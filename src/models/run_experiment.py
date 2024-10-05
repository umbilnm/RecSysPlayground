from pathlib import Path
import pandas as pd
import mlflow
from utils import TimeRangeSplit, compute_metrics, get_coo_matrix
from implicit.nearest_neighbours import CosineRecommender


PATH_TO_FILE = Path(__file__).parent


def run_experiment(n_folds: int, model_config_path: str = None):
    mlflow.set_experiment("test_experiment cosine recommender")
    interactions_df = pd.read_csv(
        PATH_TO_FILE / "../../data/interim/selected_interactions_sample.csv",
        parse_dates=["last_watch_dt"],
    )

    last_date = interactions_df["last_watch_dt"].max()
    start_date = last_date - pd.Timedelta(days=n_folds * 7)
    cv = TimeRangeSplit(start_date=start_date, periods=n_folds + 1, freq="W")
    folds_with_stats = list(
        cv.split(
            interactions_df,
            user_column="user_id",
            item_column="item_id",
            datetime_column="last_watch_dt",
            fold_stats=True,
        )
    )
    users_inv_mapping = dict(enumerate(interactions_df["user_id"].unique()))
    users_mapping = {v: k for k, v in users_inv_mapping.items()}
    top_N = 10
    items_inv_mapping = dict(enumerate(interactions_df["item_id"].unique()))
    items_mapping = {v: k for k, v in items_inv_mapping.items()}
    validation_results = pd.DataFrame()

    for fold_idx, (train_idx, test_idx, info) in enumerate(folds_with_stats):
        with mlflow.start_run(run_name=f"Fold_{fold_idx+1}"):
            train = interactions_df.loc[train_idx]
            test = interactions_df.loc[test_idx]
            train_sparse = get_coo_matrix(
                train, users_mapping=users_mapping, items_mapping=items_mapping
            ).tocsr()
            mlflow.log_param("fold_idx", fold_idx)
            mlflow.log_param("fold_start_date", info["Start date"])
            mlflow.log_param("fold_end_date", info["End date"])
            mlflow.log_param("top_N", top_N)

            model = CosineRecommender(K=20)
            model.fit(train_sparse.T)
            mlflow.log_param("model_type", "CosineRecommender")
            mlflow.log_param("K", 20)

            recs = pd.DataFrame({"user_id": interactions_df["user_id"].unique()})
            recs["item_id"] = list(model.recommend(recs["user_id"], train_sparse, N=top_N)[0])
            recs = recs.explode("item_id")
            recs["rank"] = recs.groupby("user_id").cumcount() + 1
            fold_result = compute_metrics(train, test, recs, top_N)
            for metric, value in fold_result.items():
                mlflow.log_metric(metric, value)
            validation_results = pd.concat([validation_results, fold_result], axis=1)


run_experiment(n_folds=5)
