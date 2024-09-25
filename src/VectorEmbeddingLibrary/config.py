import os
import yaml

class Config:
    def __init__(self, config_file='config.yaml'):
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

        self.config['embedding_service'] = os.getenv('EMBEDDING_SERVICE', self.config.get('embedding_service'))
        self.config['database'] = {
            'host': os.getenv('DB_HOST', self.config['database'].get('host')),
            'port': os.getenv('DB_PORT', self.config['database'].get('port')),
            'keyspace': os.getenv('DB_KEYSPACE', self.config['database'].get('keyspace'))
        }
        self.config['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        self.config['astradb'] = {
            'keyspace': os.getenv('ASTRADB_KEYSPACE'),
            'table': os.getenv('ASTRADB_TABLE'),
            'username': os.getenv('ASTRADB_USERNAME'),
            'password': os.getenv('ASTRADB_PASSWORD'),
            'secure_connect_bundle': os.getenv('ASTRADB_SECURE_CONNECT_BUNDLE')
        }

    @property
    def embedding_service(self):
        return self.config['embedding_service']

    @property
    def database(self):
        return self.config['database']

    @property
    def openai_api_key(self):
        return self.config['openai_api_key']

    @property
    def astradb(self):
        return self.config['astradb']
