import pandas as pd
import pytest

WINDOW_PATH = "datasets/forecast_windows.parquet"


@pytest.fixture(scope="module")
def windows(repo_root):
    return pd.read_parquet(repo_root / WINDOW_PATH)


def test_retrieval_candidate_boundary(windows):
    """
    A retrieval candidate's complete 14-day input + 5-day future
    must finish before the query forecast begins.

    Therefore:

        candidate_start <= query_date - 19 days
    """

    required_latest = (
        windows["query_date"]
        - pd.Timedelta(days=19)
    )

    assert (
        windows["candidate_latest_start"]
        <= required_latest
    ).all()


def test_split_dates(windows):
    train = windows[windows["split"] == "train"]
    val = windows[windows["split"] == "val"]
    test = windows[windows["split"] == "test"]

    # every split must exist, so the date checks below can never pass vacuously
    assert not train.empty and not val.empty and not test.empty

    assert train["query_date"].max() <= pd.Timestamp(
        "2015-12-31"
    )

    assert val["query_date"].min() >= pd.Timestamp(
        "2016-01-01"
    )
    assert val["query_date"].max() <= pd.Timestamp(
        "2018-12-31"
    )

    assert test["query_date"].min() >= pd.Timestamp(
        "2019-01-01"
    )


def test_no_duplicate_query_dates(windows):
    assert not windows["query_date"].duplicated().any()
