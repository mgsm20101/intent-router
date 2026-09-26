import sys
import types

import pytest

from src import classifiers
from src.schema import Prediction


def _fake_module(monkeypatch: pytest.MonkeyPatch, name: str, method: str) -> None:
    """Stand in for a classifier module so the real model code is never imported."""
    module = types.ModuleType(name)
    module.classify = lambda text: Prediction(intent="greeting", method=method)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize(
    ("method", "module_name"),
    [
        ("encoder", "src.encoder_classifier.predict"),
        ("llm", "src.llm_classifier.classifier"),
    ],
)
def test_get_returns_that_methods_classify_function(
    monkeypatch: pytest.MonkeyPatch, method: str, module_name: str
) -> None:
    _fake_module(monkeypatch, module_name, method)

    classify = classifiers.get(method)

    assert classify is sys.modules[module_name].classify
    assert classify("hi").method == method


def test_get_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown method 'regex'"):
        classifiers.get("regex")
