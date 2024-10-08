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

## PostgreSQL Setup and Usage

To configure and use PostgreSQL with the `VectorEmbeddingLibrary`, follow these steps:

### Configuration

Add the following configuration to your `config.yaml` file:

```yaml
database:
  type: 'postgresql'
  host: 'your_postgresql_host'
  port: 5432
  dbname: 'VectorEmbeddingDB'
  user: 'your_postgresql_username'
  password: 'your_postgresql_password'
```

You can also set the following environment variables to override the configuration:

- `DB_TYPE`: The database type (default: `postgresql`)
- `DB_HOST`: The PostgreSQL host (default: `localhost`)
- `DB_PORT`: The PostgreSQL port (default: `5432`)
- `DB_NAME`: The PostgreSQL database name (default: `VectorEmbeddingDB`)
- `DB_USER`: The PostgreSQL username
- `DB_PASSWORD`: The PostgreSQL password

### Usage

To use PostgreSQL for storing and querying vector embeddings, you need to initialize the database connection and perform the necessary operations. Below is an example of how to do this:

```python
import psycopg2

# Initialize the PostgreSQL connection
conn = psycopg2.connect(
    host='your_postgresql_host',
    port=5432,
    dbname='VectorEmbeddingDB',
    user='your_postgresql_username',
    password='your_postgresql_password'
)

# Create a cursor object
cur = conn.cursor()

# Example: Create a table for storing vector embeddings
cur.execute('''
CREATE TABLE IF NOT EXISTS vector_embeddings (
    id SERIAL PRIMARY KEY,
    vector FLOAT8[] NOT NULL,
    metadata JSONB
);
''')

# Commit the transaction
conn.commit()

# Example: Insert a vector embedding with metadata
vector = [0.1, 0.2, 0.3]
metadata = {'id': 'sample_id'}
cur.execute(
    'INSERT INTO vector_embeddings (vector, metadata) VALUES (%s, %s)',
    (vector, json.dumps(metadata))
)

# Commit the transaction
conn.commit()

# Example: Query similar vectors (this is a placeholder, implement your own similarity search logic)
cur.execute('SELECT * FROM vector_embeddings')
rows = cur.fetchall()
for row in rows:
    print(row)

# Close the cursor and connection
cur.close()
conn.close()
```

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

## Developer Guide

### Architecture

The library is structured into several modules:

- `embedding.py`: Contains the `VectorEmbedder` abstract base class and the `OpenAIEmbedder` concrete class for embedding text into vectors.
- `similarity_search.py`: Contains the `SimilaritySearch` abstract base class and the `AstraDBSimilaritySearch` concrete class for performing similarity searches.
- `config.py`: Handles the configuration of the library using a `config.yaml` file and environment variables.
- `main.py`: The main entry point for running the vector embedding and similarity search process.

### Coding Standards

- Follow PEP 8 guidelines for Python code style.
- Write clear and concise docstrings for all classes, methods, and functions.
- Use meaningful variable and function names.
- Prioritize code readability and maintainability.

### Testing Procedures

- Write unit tests for all new functions and classes.
- Use `pytest` as the testing framework.
- Mock external dependencies and API calls in tests.
- Test both normal scenarios and edge cases.
- Use descriptive names for test functions.
- Ensure tests cover the specific functionality implemented.
