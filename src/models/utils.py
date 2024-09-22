import pytest
from typing import List, Dict, Callable
from more_itertools import pairwise
import scipy.sparse as sp
import numpy as np
import pandas as pd
from implicit.nearest_neighbours import ItemItemRecommender

class Evaluator:
    def __init__(self, top_k: List[int], metrics: List[str], predicted_col: str, true_col: str):
        self.result = {}
        self.top_k = top_k
        self.metrics = metrics
        self.predicted_col = predicted_col
        self.true_col = true_col
        
        self.metrics_mapper: Dict[str, Callable] = {
            'NDCG': self._calc_ndcg,
            'Precision': self._calc_precision,
            'Recall': self._calc_recall,
            'MAP': self._calc_map
        }
        
        for metric in metrics:
            if metric not in self.metrics_mapper:
                raise ValueError(f"Metric '{metric}' is not implemented.")

    def _get_top_k(self, items: List[int], k: int) -> List[int]:
        return items[:k]

    def _calc_ndcg_sample(self, true_sample: List[int], predicted_sample: List[int], k: int) -> float:
        dcg = sum((1 / np.log2(i + 2)) for i, p in enumerate(predicted_sample[:k]) if p in true_sample)
        idcg = sum((1 / np.log2(i + 2)) for i in range(min(k, len(true_sample))))
        return dcg / idcg if idcg > 0 else 0

    def _calc_ndcg(self, true_items: List[List[int]], predicted_items: List[List[int]]) -> Dict[int, float]:
        return {
            k: np.mean([self._calc_ndcg_sample(t, p, k) for t, p in zip(true_items, predicted_items)])
            for k in self.top_k
        }

    def _calc_precision_sample(self, true_sample: List[int], predicted_sample: List[int], k: int) -> float:
        top_k_predicted = self._get_top_k(predicted_sample, k)
        if k == 0:
            return 0
        return len(set(top_k_predicted) & set(true_sample)) / k

    def _calc_precision(self, true_items: List[List[int]], predicted_items: List[List[int]]) -> Dict[int, List[float]]:
        return {
            k: np.mean([self._calc_precision_sample(t, p, k) for t, p in zip(true_items, predicted_items)])
            for k in self.top_k
        }

    def _calc_recall_sample(self, true_sample: List[int], predicted_sample: List[int], k: int) -> float:
        top_k_predicted = self._get_top_k(predicted_sample, k)
        return len(set(top_k_predicted) & set(true_sample)) / len(true_sample)

    def _calc_recall(self, true_items: List[List[int]], predicted_items: List[List[int]]) -> Dict[int, List[float]]:
        return {
            k: np.mean([self._calc_recall_sample(t, p, k) for t, p in zip(true_items, predicted_items)])
            for k in self.top_k
        }
    
    def _calc_avg_precision_sample(self, true_sample: List[int], predicted_sample: List[int], k: int) -> float:
        score = 0
        relevant_count = 0
        for i in range(min(len(predicted_sample), k)):
            if predicted_sample[i] in true_sample:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                score += precision_at_i
        
        # Если не было релевантных элементов, возвращаем 0
        return score / min(len(true_sample), k) if true_sample else 0

    def _calc_map(self, true_items: List[List[int]], predicted_items: List[List[int]]) -> Dict[int, float]:
        return {
            k: np.mean([self._calc_avg_precision_sample(t, p, k) for t, p in zip(true_items, predicted_items)])
            for k in self.top_k
        }

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Dict[int, List[float]]]:
        self.result = {}
        true_items = df[self.true_col].tolist()
        predicted_items = df[self.predicted_col].tolist()

        for metric in self.metrics:
            metric_results = self.metrics_mapper[metric](true_items, predicted_items)
            self.result[metric] = metric_results

        return self.result


class TimeRangeSplit():
    """
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html
    """
    def __init__(self, 
                 start_date, 
                 end_date=None, 
                 freq='D', 
                 periods=None, 
                 tz=None, 
                 normalize=False, 
                 closed=None, 
                 train_min_date=None,
                 filter_cold_users=True, 
                 filter_cold_items=True, 
                 filter_already_seen=True):
        
        self.start_date = start_date
        if end_date is None and periods is None:
            raise ValueError('Either "end_date" or "periods" must be non-zero, not both at the same time.')

        self.end_date = end_date
        self.freq = freq
        self.periods = periods
        self.tz = tz
        self.normalize = normalize
        self.closed = closed
        self.train_min_date = pd.to_datetime(train_min_date, errors='raise')
        self.filter_cold_users = filter_cold_users
        self.filter_cold_items = filter_cold_items
        self.filter_already_seen = filter_already_seen

        self.date_range = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq=freq, 
            periods=periods, 
            tz=tz, 
            normalize=normalize, 
            closed=closed)

        self.max_n_splits = max(0, len(self.date_range) - 1)
        if self.max_n_splits == 0:
            raise ValueError('Provided parametrs set an empty date range.') 

    def split(self, 
              df, 
              user_column='user_id',
              item_column='item_id',
              datetime_column='date',
              fold_stats=False):
        df_datetime = df[datetime_column]
        if self.train_min_date is not None:
            train_min_mask = df_datetime >= self.train_min_date
        else:
            train_min_mask = df_datetime.notnull()

        date_range = self.date_range[(self.date_range >= df_datetime.min()) & 
                                     (self.date_range <= df_datetime.max())]

        for start, end in pairwise(date_range):
            fold_info = {
                'Start date': start,
                'End date': end
            }
            train_mask = train_min_mask & (df_datetime < start)
            train_idx = df.index[train_mask]
            if fold_stats:
                fold_info['Train'] = len(train_idx)

            test_mask = (df_datetime >= start) & (df_datetime < end)
            test_idx = df.index[test_mask]
            
            if self.filter_cold_users:
                new = np.setdiff1d(
                    df.loc[test_idx, user_column].unique(), 
                    df.loc[train_idx, user_column].unique())
                new_idx = df.index[test_mask & df[user_column].isin(new)]
                test_idx = np.setdiff1d(test_idx, new_idx)
                test_mask = df.index.isin(test_idx)
                if fold_stats:
                    fold_info['New users'] = len(new)
                    fold_info['New users interactions'] = len(new_idx)

            if self.filter_cold_items:
                new = np.setdiff1d(
                    df.loc[test_idx, item_column].unique(), 
                    df.loc[train_idx, item_column].unique())
                new_idx = df.index[test_mask & df[item_column].isin(new)]
                test_idx = np.setdiff1d(test_idx, new_idx)
                test_mask = df.index.isin(test_idx)
                if fold_stats:
                    fold_info['New items'] = len(new)
                    fold_info['New items interactions'] = len(new_idx)

            if self.filter_already_seen:
                user_item = [user_column, item_column]
                train_pairs = df.loc[train_idx, user_item].set_index(user_item).index
                test_pairs = df.loc[test_idx, user_item].set_index(user_item).index
                intersection = train_pairs.intersection(test_pairs)
                print(f'Already seen number: {len(intersection)}')
                test_idx = test_idx[~test_pairs.isin(intersection)]
                # test_mask = rd.df.index.isin(test_idx)
                if fold_stats:
                    fold_info['Known interactions'] = len(intersection)

            if fold_stats:
                fold_info['Test'] = len(test_idx)

            yield (train_idx, test_idx, fold_info)

    def get_n_splits(self, df, datetime_column='date'):
        df_datetime = df[datetime_column]
        if self.train_min_date is not None:
            df_datetime = df_datetime[df_datetime >= self.train_min_date]

        date_range = self.date_range[(self.date_range >= df_datetime.min()) & 
                                     (self.date_range <= df_datetime.max())]

        return max(0, len(date_range) - 1)


def calculate_novelty(train_interactions:pd.DataFrame, recommendations:pd.DataFrame, top_n): 
    users = recommendations['user_id'].unique()
    n_users = train_interactions['user_id'].nunique()
    n_users_per_item = train_interactions.groupby('item_id')['user_id'].nunique()

    recommendations = recommendations.loc[recommendations['rank'] <= top_n].copy()
    recommendations['n_users_per_item'] = recommendations['item_id'].map(n_users_per_item)
    recommendations['n_users_per_item'] = recommendations['n_users_per_item'].fillna(1)
    recommendations['item_novelty'] = -np.log2(recommendations['n_users_per_item'] / n_users)

    item_novelties = recommendations[['user_id', 'rank', 'item_novelty']]
    
    miuf_at_k = item_novelties.loc[item_novelties['rank'] <= top_n, ['user_id', 'item_novelty']]
    miuf_at_k = miuf_at_k.groupby('user_id').agg('mean').squeeze()

    return miuf_at_k.reindex(users).mean()

def compute_metrics(train:pd.DataFrame, test:pd.DataFrame, recs:pd.DataFrame, top_N:int) -> pd.Series:
    result = {}
    test_recs = test.set_index(['user_id', 'item_id']).join(recs.set_index(['user_id', 'item_id']))
    test_recs = test_recs.sort_values(by=['user_id', 'rank'])

    test_recs['users_item_count'] = test_recs.groupby(level='user_id')['rank'].transform(np.size)
    test_recs['reciprocal_rank'] = (1 / test_recs['rank']).fillna(0)
    test_recs['cumulative_rank'] = test_recs.groupby(level='user_id').cumcount() + 1
    test_recs['cumulative_rank'] = test_recs['cumulative_rank'] / test_recs['rank']
    
    users_count = test_recs.index.get_level_values('user_id').nunique()
    
    for k in range(1, top_N + 1):
        hit_k = f'hit@{k}'
        test_recs[hit_k] = test_recs['rank'] <= k
        result[f'Precision@{k}'] = (test_recs[hit_k] / k).sum() / users_count
        result[f'Recall@{k}'] = (test_recs[hit_k] / test_recs['users_item_count']).sum() / users_count
        
    result[f'MAP@{top_N}'] = (test_recs['cumulative_rank'] / test_recs['users_item_count']).sum() / users_count
    result[f'Novelty@{top_N}'] = calculate_novelty(train, recs, top_N)
    
    return pd.Series(result)

def get_coo_matrix(df, 
                   user_col='user_id', 
                   item_col='item_id', 
                   weight_col=None, 
                   users_mapping={}, 
                   items_mapping={}):
    
    if weight_col is None:
        weights = np.ones(len(df), dtype=np.float32)
    else:
        weights = df[weight_col].astype(np.float32)

    interaction_matrix = sp.coo_matrix((
        weights, 
        (
            df[user_col].map(users_mapping.get), 
            df[item_col].map(items_mapping.get)
        )
    ))
    return interaction_matrix



def test_map_perfect_match():
    evaluator = Evaluator(top_k=[5], metrics=['MAP'], predicted_col='predicted', true_col='true')
    df = pd.DataFrame({
        'true': [[1, 2, 3, 4, 5]],
        'predicted': [[1, 2, 3, 4, 5]]
    })
    result = evaluator.evaluate(df)
    assert result['MAP'][5] == pytest.approx(1.0), "MAP должен быть 1.0 для идеального совпадения"

def test_map_partial_match():
    evaluator = Evaluator(top_k=[5], metrics=['MAP'], predicted_col='predicted', true_col='true')
    df = pd.DataFrame({
        'true': [[1, 2, 3, 4, 5]],
        'predicted': [[1, 2, 6, 7, 8]]
    })
    result = evaluator.evaluate(df)
    assert result['MAP'][5] == pytest.approx((1/1 + 2/2)/5), "MAP должен быть около 0.3 для частичного совпадения"

def test_map_no_match():
    evaluator = Evaluator(top_k=[5], metrics=['MAP'], predicted_col='predicted', true_col='true')
    df = pd.DataFrame({
        'true': [[1, 2, 3, 4, 5]],
        'predicted': [[6, 7, 8, 9, 10]]
    })
    result = evaluator.evaluate(df)
    assert result['MAP'][5] == pytest.approx(0.0), "MAP должен быть 0.0 при отсутствии совпадений"

def test_map_more_predictions_than_true():
    evaluator = Evaluator(top_k=[5], metrics=['MAP'], predicted_col='predicted', true_col='true')
    df = pd.DataFrame({
        'true': [[1, 2]],
        'predicted': [[1, 2, 3, 4, 5]]
    })
    result = evaluator.evaluate(df)
    assert result['MAP'][5] == pytest.approx(1.0), "MAP должен быть 1.0, если все истинные элементы предсказаны первыми"

def test_map_with_fewer_predictions():
    evaluator = Evaluator(top_k=[5], metrics=['MAP'], predicted_col='predicted', true_col='true')
    df = pd.DataFrame({
        'true': [[1, 2, 3, 4, 5]],
        'predicted': [[1, 2]]
    })
    result = evaluator.evaluate(df)
    assert result['MAP'][5] == pytest.approx((1/1 + 2/2)/5), "MAP должен быть меньше 1.0 при меньшем количестве предсказанных элементов"

def test_map_at_different_top_k():
    evaluator = Evaluator(top_k=[1, 3, 5], metrics=['MAP'], predicted_col='predicted', true_col='true')
    df = pd.DataFrame({
        'true': [[1, 2, 3, 4, 5]],
        'predicted': [[1, 2, 6, 7, 8]]
    })
    result = evaluator.evaluate(df)
    assert result['MAP'][1] == pytest.approx(1.0), "MAP@1 должен быть 1.0"
    assert result['MAP'][3] == pytest.approx((1/1 + 2/2) / 3), "MAP@3 должен быть около 0.833"
    assert result['MAP'][5] == pytest.approx((1/1 + 2/2) / 5), "MAP@5 должен быть около 0.4"

def test_map_with_multiple_samples():
    evaluator = Evaluator(top_k=[5], metrics=['MAP'], predicted_col='predicted', true_col='true')
    df = pd.DataFrame({
        'true': [[1, 2, 3, 4, 5], [1, 2, 3]],
        'predicted': [[1, 2, 6, 7, 8], [1, 4, 5, 6, 7]]
    })
    result = evaluator.evaluate(df)
    expected_map = ((1/1 + 2/2) / 5 + (1/1) / 5) / 2
    assert result['MAP'][5] == pytest.approx(expected_map), "MAP должен корректно вычисляться для нескольких пользователей"


def generate_implicit_recs_mapper(
    model: ItemItemRecommender,
    train_matrix: sp.csr_matrix,
    top_N: int,
    user_mapping: dict,
    item_inv_mapping: dict,
    filter_already_liked_items: bool = True
):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.recommend(user_id, 
                               train_matrix.tocsr(), 
                               N=top_N, 
                               filter_already_liked_items=filter_already_liked_items)
        return [item_inv_mapping[item] for item in recs[0]]
    return _recs_mapper