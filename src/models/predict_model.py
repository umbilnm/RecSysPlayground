import os
import click
import logging
from pathlib import Path
import json
import pandas as pd
import scipy.sparse as sp
from implicit.nearest_neighbours import CosineRecommender
from .utils import generate_implicit_recs_mapper


@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("path_to_save")
def inference_implicit_model(
    data_path: str, model_path: str, path_to_save: str
) -> None:
    logger = logging.getLogger(__name__)
    model = CosineRecommender(K=20)
    model = model.load(model_path)
    logger.info(f"model loaded from {model_path}")
    interactions_train_matrix = sp.load_npz(
        os.path.join(data_path, "interactions_matrix.npz")
    )

    with open(os.path.join(data_path, "users_mapping.json"), "r") as f:
        user_mapping = json.load(f)

    with open(os.path.join(data_path, "items_inv_mapping.json"), "r") as f:
        item_inv_mapping = json.load(f)

    recs_mapper = generate_implicit_recs_mapper(
        model=model,
        train_matrix=interactions_train_matrix.tocsr(),
        top_N=10,
        user_mapping=user_mapping,
        item_inv_mapping=item_inv_mapping,
    )

    train_data = pd.read_csv(
        "/home/umbilnm/RecSysPlayground/data/interim/one_week_train.csv"
    )
    users = (
        pd.Series(train_data["user_id"].unique()).to_frame(name="user_id").astype(str)
    )
    logger.info("inferencing")
    recs = users["user_id"].apply(recs_mapper)
    users["recs"] = recs

    users.to_csv(path_to_save, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    inference_implicit_model()
