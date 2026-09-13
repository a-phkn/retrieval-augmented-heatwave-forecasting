
import pandas as pd


WINDOW_PATH = "datasets/forecast_windows.parquet"


def test_retrieval_candidate_boundary():
    """
    A retrieval candidate's complete 14-day input + 5-day future
    must finish before the query forecast begins.

    Therefore:

        candidate_start <= query_date - 19 days
    """

    windows = pd.read_parquet(WINDOW_PATH)

    required_latest = (
        windows["query_date"]
        - pd.Timedelta(days=19)
    )

    assert (
        windows["candidate_latest_start"]
        <= required_latest
    ).all()


def test_split_dates():
    windows = pd.read_parquet(WINDOW_PATH)

    train = windows[windows["split"] == "train"]
    val = windows[windows["split"] == "val"]
    test = windows[windows["split"] == "test"]

    if not train.empty:
        assert train["query_date"].max() <= pd.Timestamp(
            "2015-12-31"
        )

    if not val.empty:
        assert val["query_date"].min() >= pd.Timestamp(
            "2016-01-01"
        )
        assert val["query_date"].max() <= pd.Timestamp(
            "2018-12-31"
        )

    if not test.empty:
        assert test["query_date"].min() >= pd.Timestamp(
            "2019-01-01"
        )


def test_no_duplicate_query_dates():
    windows = pd.read_parquet(WINDOW_PATH)

    assert not windows["query_date"].duplicated().any()
