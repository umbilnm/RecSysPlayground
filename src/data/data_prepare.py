# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
import scipy.sparse as sp
import numpy as np


@click.group()
def cli():
    """Group of commands for data preparation."""
    pass

@cli.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('train', type=click.Path())
@click.argument('test', type=click.Path())
def split_data(input, train, test):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('splitting data by time')
    data = pd.read_csv(input, parse_dates=["last_watch_dt"])
    gap = pd.Timedelta(days=2)
    start_test_date = data['last_watch_dt'].max() - pd.Timedelta(days=7)
    end_test_date = data['last_watch_dt'].max()
    start_train_date = data['last_watch_dt'].max() - pd.Timedelta(days=14) - gap
    end_train_date = data['last_watch_dt'].max() - gap - pd.Timedelta(days=7)

    ## splitting 
    train_sample = data[(data['last_watch_dt'] > start_train_date) & (data['last_watch_dt'] < end_train_date)]
    test_sample = data[(data['last_watch_dt'] > start_test_date) & (data['last_watch_dt'] < end_test_date)]

    ## saving     
    train_sample.to_csv(train, index=False)
    test_sample.to_csv(test, index=False)


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def get_coo_matrix(input_path: str, output_path: str, user_col='user_id', item_col='item_id', weight_col=None):
    train_sample = pd.read_csv(input_path)
    logger = logging.getLogger(__name__)
    logger.info('prepare interactions coo_matrix')

    users_inv_mapping = dict(enumerate(train_sample['user_id'].unique()))
    users_mapping = {v: k for k, v in users_inv_mapping.items()}
    items_inv_mapping = dict(enumerate(train_sample['item_id'].unique()))
    items_mapping = {v: k for k, v in items_inv_mapping.items()}
    if weight_col is None:
        weights = np.ones(len(train_sample), dtype=np.float32)
    else:
        weights = train_sample[weight_col].astype(np.float32)

    interaction_matrix = sp.coo_matrix((
    weights, 
    (
        train_sample[user_col].map(users_mapping.get), 
        train_sample[item_col].map(items_mapping.get)
    )
))
    sp.save_npz(output_path, interaction_matrix)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    cli()