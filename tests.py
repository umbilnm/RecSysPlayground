import pytest
import pandas as pd
from src.models.utils import Evaluator


@pytest.fixture
def evaluator():
    return Evaluator(
        top_k=[2],
        metrics=["NDCG", "Precision", "Recall", "MAP"],
        predicted_col="predicted_col",
        true_col="true_col",
    )


def test_specific_sample(evaluator):
    data = {"true_col": [[1, 2, 3], [4, 5, 6]], "predicted_col": [[1, 3, 2], [6, 4, 5]]}

    df = pd.DataFrame(data)

    result = evaluator.evaluate(df)

    expected_ndcg = 1.0
    expected_precision = 1.0
    expected_recall = 2 / 3
    expected_map = 1.0
    assert (
        round(result["NDCG"][2], 2) == expected_ndcg
    ), f"NDCG@2 is incorrect. Expected {expected_ndcg}, got {result['NDCG'][2]}"

    assert (
        round(result["Precision"][2], 2) == expected_precision
    ), f"Precision@2 is incorrect. Expected {expected_precision}, got {result['Precision'][2]}"

    assert round(result["Recall"][2], 2) == round(
        expected_recall, 2
    ), f"Recall@2 is incorrect. Expected {expected_recall}, got {result['Recall'][2]}"

    assert (
        round(result["MAP"][2], 2) == expected_map
    ), f"MAP@2 is incorrect. Expected {expected_map}, got {result['MAP'][2]}"


def test_recall_perfect_match():
    evaluator = Evaluator(top_k=[5], metrics=["Recall"], predicted_col="predicted", true_col="true")
    df = pd.DataFrame({"true": [[1, 2, 3, 4, 5]], "predicted": [[1, 2, 3, 4, 5]]})
    result = evaluator.evaluate(df)
    assert result["Recall"][5] == pytest.approx(
        1.0
    ), "Recall должен быть 1.0 для идеального совпадения"


def test_recall_partial_match():
    evaluator = Evaluator(top_k=[5], metrics=["Recall"], predicted_col="predicted", true_col="true")
    df = pd.DataFrame({"true": [[1, 2, 3, 4, 5]], "predicted": [[1, 2, 6, 7, 8]]})
    result = evaluator.evaluate(df)
    assert result["Recall"][5] == pytest.approx(
        0.4
    ), "Recall должен быть 0.4, если совпало 2 из 5 элементов"


def test_recall_no_match():
    evaluator = Evaluator(top_k=[5], metrics=["Recall"], predicted_col="predicted", true_col="true")
    df = pd.DataFrame({"true": [[1, 2, 3, 4, 5]], "predicted": [[6, 7, 8, 9, 10]]})
    result = evaluator.evaluate(df)
    assert result["Recall"][5] == pytest.approx(
        0.0
    ), "Recall должен быть 0.0 при отсутствии совпадений"


def test_recall_more_predictions_than_true():
    evaluator = Evaluator(top_k=[5], metrics=["Recall"], predicted_col="predicted", true_col="true")
    df = pd.DataFrame({"true": [[1, 2]], "predicted": [[1, 2, 3, 4, 5]]})
    result = evaluator.evaluate(df)
    assert result["Recall"][5] == pytest.approx(
        1.0
    ), "Recall должен быть 1.0, если все истинные элементы предсказаны, даже если есть лишние"


def test_recall_at_different_top_k():
    evaluator = Evaluator(
        top_k=[1, 3, 5], metrics=["Recall"], predicted_col="predicted", true_col="true"
    )
    df = pd.DataFrame({"true": [[1, 2, 3, 4, 5]], "predicted": [[1, 2, 3, 6, 7]]})
    result = evaluator.evaluate(df)
    assert result["Recall"][1] == pytest.approx(0.2), "Recall@1 должен быть 0.2"
    assert result["Recall"][3] == pytest.approx(0.6), "Recall@3 должен быть 0.6"
    assert result["Recall"][5] == pytest.approx(
        0.6
    ), "Recall@5 должен быть 0.6, так как только 3 элемента из 5 истинных угаданы"


def test_map_perfect_match():
    evaluator = Evaluator(top_k=[5], metrics=["MAP"], predicted_col="predicted", true_col="true")
    df = pd.DataFrame({"true": [[1, 2, 3, 4, 5]], "predicted": [[1, 2, 3, 4, 5]]})
    result = evaluator.evaluate(df)
    assert result["MAP"][5] == pytest.approx(1.0), "MAP должен быть 1.0 для идеального совпадения"


def test_map_partial_match():
    evaluator = Evaluator(top_k=[5], metrics=["MAP"], predicted_col="predicted", true_col="true")
    df = pd.DataFrame({"true": [[1, 2, 3, 4, 5]], "predicted": [[1, 2, 6, 7, 8]]})
    result = evaluator.evaluate(df)
    assert result["MAP"][5] == pytest.approx(
        (1 / 1 + 2 / 2) / 5
    ), "MAP должен быть около 0.3 для частичного совпадения"


def test_map_no_match():
    evaluator = Evaluator(top_k=[5], metrics=["MAP"], predicted_col="predicted", true_col="true")
    df = pd.DataFrame({"true": [[1, 2, 3, 4, 5]], "predicted": [[6, 7, 8, 9, 10]]})
    result = evaluator.evaluate(df)
    assert result["MAP"][5] == pytest.approx(0.0), "MAP должен быть 0.0 при отсутствии совпадений"


def test_map_more_predictions_than_true():
    evaluator = Evaluator(top_k=[5], metrics=["MAP"], predicted_col="predicted", true_col="true")
    df = pd.DataFrame({"true": [[1, 2]], "predicted": [[1, 2, 3, 4, 5]]})
    result = evaluator.evaluate(df)
    assert result["MAP"][5] == pytest.approx(
        1.0
    ), "MAP должен быть 1.0, если все истинные элементы предсказаны первыми"


def test_map_with_fewer_predictions():
    evaluator = Evaluator(top_k=[5], metrics=["MAP"], predicted_col="predicted", true_col="true")
    df = pd.DataFrame({"true": [[1, 2, 3, 4, 5]], "predicted": [[1, 2]]})
    result = evaluator.evaluate(df)
    assert result["MAP"][5] == pytest.approx(
        (1 / 1 + 2 / 2) / 5
    ), "MAP должен быть меньше 1.0 при меньшем количестве предсказанных элементов"


def test_map_at_different_top_k():
    evaluator = Evaluator(
        top_k=[1, 3, 5], metrics=["MAP"], predicted_col="predicted", true_col="true"
    )
    df = pd.DataFrame({"true": [[1, 2, 3, 4, 5]], "predicted": [[1, 2, 6, 7, 8]]})
    result = evaluator.evaluate(df)
    assert result["MAP"][1] == pytest.approx(1.0), "MAP@1 должен быть 1.0"
    assert result["MAP"][3] == pytest.approx((1 / 1 + 2 / 2) / 3), "MAP@3 должен быть около 0.833"
    assert result["MAP"][5] == pytest.approx((1 / 1 + 2 / 2) / 5), "MAP@5 должен быть около 0.4"


def test_map_with_multiple_samples():
    evaluator = Evaluator(top_k=[5], metrics=["MAP"], predicted_col="predicted", true_col="true")
    df = pd.DataFrame(
        {
            "true": [[1, 2, 3, 4, 5], [1, 2, 3]],
            "predicted": [[1, 2, 6, 7, 8], [1, 4, 5, 6, 7]],
        }
    )
    result = evaluator.evaluate(df)
    expected_map = ((1 / 1 + 2 / 2) / 5 + (1 / 1) / 5) / 2
    assert result["MAP"][5] == pytest.approx(
        expected_map
    ), "MAP должен корректно вычисляться для нескольких пользователей"


# 1) 1 +
