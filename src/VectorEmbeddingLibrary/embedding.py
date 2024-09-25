from abc import ABC, abstractmethod
import openai

class VectorEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list:
        pass

class OpenAIEmbedder(VectorEmbedder):
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key

    def embed_text(self, text: str) -> list:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return response['data'][0]['embedding']
