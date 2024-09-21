# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('train', type=click.Path())
@click.argument('test', type=click.Path())
def split_data(input, train, test):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
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


@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.argument('out', type=click.Path())
def get_coo_matrix(input, out):
    df = pd.read_csv(input_path)
    users_inv_mapping = dict(enumerate(train_sample['user_id'].unique()))
    users_mapping = {v: k for k, v in users_inv_mapping.items()}
    top_N = 10
    last_n_days = 7
    items_inv_mapping = dict(enumerate(train_sample['item_id'].unique()))
    items_mapping = {v: k for k, v in items_inv_mapping.items()}
    interaction_matrix = sp.coo_matrix((
    weights, 
    (
        df[user_col].map(users_mapping.get), 
        df[item_col].map(items_mapping.get)
    )
))
    sp.save_npz(out, interaction_matrix)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    split_data()