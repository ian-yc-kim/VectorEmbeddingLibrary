import pytest
from embedding import OpenAIEmbedder


class MockOpenAI:
    @staticmethod
    def create(input, model):
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}


@pytest.fixture
def mock_openai(monkeypatch):
    monkeypatch.setattr("openai.Embedding", MockOpenAI)


@pytest.fixture
def embedder():
    return OpenAIEmbedder(api_key="test_key")


def test_embed_text(mock_openai, embedder):
    text = "test text"
    expected_embedding = [0.1, 0.2, 0.3]
    embedding = embedder.embed_text(text)
    assert embedding == expected_embedding


def test_embed_empty_text(mock_openai, embedder):
    text = ""
    expected_embedding = [0.1, 0.2, 0.3]
    embedding = embedder.embed_text(text)
    assert embedding == expected_embedding


def test_embed_special_characters(mock_openai, embedder):
    text = "!@#$%^&*()"
    expected_embedding = [0.1, 0.2, 0.3]
    embedding = embedder.embed_text(text)
    assert embedding == expected_embedding


def test_embed_long_text(mock_openai, embedder):
    text = "a" * 1000
    expected_embedding = [0.1, 0.2, 0.3]
    embedding = embedder.embed_text(text)
    assert embedding == expected_embedding
