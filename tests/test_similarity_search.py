import pytest
from VectorEmbeddingLibrary.similarity_search import AstraDBSimilaritySearch
from unittest.mock import patch, MagicMock

@pytest.fixture
@patch('VectorEmbeddingLibrary.similarity_search.Cluster')
def astra_db_search(mock_cluster):
    mock_session = MagicMock()
    mock_cluster.return_value.connect.return_value = mock_session
    search = AstraDBSimilaritySearch(keyspace='test_keyspace', table='test_table', username='user', password='pass', secure_connect_bundle='bundle.zip')
    return search


def test_index_vector(astra_db_search):
    vector = [0.1, 0.2, 0.3]
    metadata = {'id': '123'}
    astra_db_search.index_vector(vector, metadata)
    astra_db_search.session.execute.assert_called_once_with(f"INSERT INTO test_keyspace.test_table (id, vector) VALUES (%s, %s)", (metadata['id'], vector))


def test_query_similar(astra_db_search):
    vector = [0.1, 0.2, 0.3]
    top_k = 5
    mock_rows = [
        MagicMock(id='1', vector=[0.1, 0.2, 0.3]),
        MagicMock(id='2', vector=[0.4, 0.5, 0.6]),
        MagicMock(id='3', vector=[0.7, 0.8, 0.9])
    ]
    astra_db_search.session.execute.return_value = mock_rows
    results = astra_db_search.query_similar(vector, top_k)
    expected_results = [('1', 0.0), ('2', 0.5196152422706632), ('3', 1.0392304845413265)]
    assert results == expected_results[:top_k]
