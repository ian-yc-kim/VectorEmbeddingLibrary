# VectorEmbeddingLibrary

## Overview

The `VectorEmbeddingLibrary` is a Python library designed to convert text inputs into vector embeddings and perform similarity searches by querying a Database-as-a-Service (DBaaS). It integrates OpenAI's vector embedding service and DataStax AstraDB for similarity search.

## Installation

To install the library, you need to have [Poetry](https://python-poetry.org/) installed. Then, you can install the dependencies by running:

```bash
poetry install
```

## Configuration

The library can be configured using a `config.yaml` file and environment variables. Below is an example of the `config.yaml` file:

```yaml
embedding_service: 'openai'
database:
  host: 'localhost'
  port: 9042
  keyspace: 'VectorEmbeddingDB'
```

You can also set the following environment variables to override the configuration:

- `EMBEDDING_SERVICE`: The embedding service to use (default: `openai`)
- `DB_HOST`: The database host (default: `localhost`)
- `DB_PORT`: The database port (default: `9042`)
- `DB_KEYSPACE`: The database keyspace (default: `VectorEmbeddingDB`)
- `OPENAI_API_KEY`: The API key for OpenAI
- `ASTRADB_KEYSPACE`: The keyspace for AstraDB
- `ASTRADB_TABLE`: The table for AstraDB
- `ASTRADB_USERNAME`: The username for AstraDB
- `ASTRADB_PASSWORD`: The password for AstraDB
- `ASTRADB_SECURE_CONNECT_BUNDLE`: The path to the secure connect bundle for AstraDB

## Usage

### OpenAIEmbedder

The `OpenAIEmbedder` class is used to convert text into vector embeddings using OpenAI's embedding service.

```python
from VectorEmbeddingLibrary.embedding import OpenAIEmbedder

# Initialize the embedder with your OpenAI API key
embedder = OpenAIEmbedder(api_key='your_openai_api_key')

# Embed text into a vector
text = 'Hello, world!'
vector = embedder.embed_text(text)
print(vector)
```

### AstraDBSimilaritySearch

The `AstraDBSimilaritySearch` class is used to perform similarity searches using DataStax AstraDB.

```python
from VectorEmbeddingLibrary.similarity_search import AstraDBSimilaritySearch

# Initialize the similarity search with your AstraDB credentials
similarity_search = AstraDBSimilaritySearch(
    keyspace='your_keyspace',
    table='your_table',
    username='your_username',
    password='your_password',
    host='your_host',
    port=9042,
    secure_connect_bundle='path_to_secure_connect_bundle'
)

# Index a vector with associated metadata
vector = [0.1, 0.2, 0.3]
metadata = {'id': 'sample_id'}
similarity_search.index_vector(vector, metadata)

# Query similar vectors
top_k = 5
similar_vectors = similarity_search.query_similar(vector, top_k)
print(similar_vectors)
```

## Running Unit Tests

To run the unit tests, you can use the following command:

```bash
poetry run pytest
```

This will execute all the tests in the `tests` directory.
