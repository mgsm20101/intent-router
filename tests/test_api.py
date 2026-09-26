import pytest
from fastapi.testclient import TestClient

from src import classifiers
from src.api import main
from src.schema import Prediction


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    def fake_get(method: str):
        return lambda text: Prediction(
            intent="refund_request", confidence=0.9, latency_ms=1.0, method=method
        )

    monkeypatch.setattr(classifiers, "get", fake_get)
    return TestClient(main.app)


def test_classify_returns_the_prediction(client: TestClient) -> None:
    response = client.post("/classify", json={"text": "I want my money back", "method": "llm"})

    assert response.status_code == 200
    assert response.json() == {
        "intent": "refund_request",
        "confidence": 0.9,
        "latency_ms": 1.0,
        "method": "llm",
    }


@pytest.mark.parametrize(
    "body",
    [
        {"text": "", "method": "encoder"},
        {"text": "hello", "method": "regex"},
        {"method": "encoder"},
    ],
)
def test_classify_rejects_invalid_requests(client: TestClient, body: dict) -> None:
    assert client.post("/classify", json=body).status_code == 422


def test_missing_model_maps_to_503(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(method: str):
        def classify(text: str) -> Prediction:
            raise FileNotFoundError("No trained encoder")

        return classify

    monkeypatch.setattr(classifiers, "get", missing)

    response = TestClient(main.app).post("/classify", json={"text": "hi"})

    assert response.status_code == 503
