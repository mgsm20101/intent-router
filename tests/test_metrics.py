import pytest

from src.eval.metrics import accuracy, cost_per_1k, macro_f1, percentile


def test_accuracy_is_the_fraction_of_matching_labels() -> None:
    assert accuracy(["a", "b", "c", "d"], ["a", "b", "x", "d"]) == pytest.approx(0.75)


def test_macro_f1_is_one_for_a_perfect_prediction() -> None:
    assert macro_f1(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)


def test_macro_f1_averages_per_class_scores() -> None:
    # class a: P=1, R=0.5 -> F1=2/3 ; class b: P=0.5, R=1 -> F1=2/3
    assert macro_f1(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(2 / 3)


def test_percentile_interpolates_linearly() -> None:
    values = [40.0, 10.0, 30.0, 20.0]

    assert percentile(values, 0) == 10.0
    assert percentile(values, 50) == pytest.approx(25.0)
    assert percentile(values, 100) == 40.0


def test_percentile_of_empty_list_is_zero() -> None:
    assert percentile([], 95) == 0.0


def test_cost_per_1k_is_average_seconds_times_hourly_rate() -> None:
    # 1 s per request at $3.60/h -> $0.001 per request -> $1.00 per 1k
    assert cost_per_1k([500.0, 1500.0], usd_per_hour=3.6) == pytest.approx(1.0)


def test_cost_per_1k_of_no_requests_is_zero() -> None:
    assert cost_per_1k([], usd_per_hour=0.9) == 0.0
