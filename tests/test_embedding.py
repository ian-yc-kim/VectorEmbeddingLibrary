import pytest
from VectorEmbeddingLibrary.embedding import OpenAIEmbedder
from unittest.mock import patch

@pytest.fixture
def openai_embedder():
    return OpenAIEmbedder(api_key="test_api_key")

@patch('openai.Embedding.create')
def test_embed_text(mock_create, openai_embedder):
    mock_create.return_value = {
        'data': [
            {'embedding': [0.1, 0.2, 0.3]}
        ]
    }
    text = "Hello, world!"
    embedding = openai_embedder.embed_text(text)
    assert embedding == [0.1, 0.2, 0.3]
    mock_create.assert_called_once_with(input=text, model="text-embedding-ada-002")
