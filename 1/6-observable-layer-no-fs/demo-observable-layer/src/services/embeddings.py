from src.settings import SETTINGS
from langchain_openai import OpenAIEmbeddings


class EmbeddingService:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(
            model=SETTINGS.OPENAI_EMBEDDING_MODEL,
            api_key=SETTINGS.OPENAI_API_KEY,
            base_url=SETTINGS.OPENAI_BASE_URL or None,
        )

    def embed_text(self, text: str):
        """Embed a single text."""
        return self.embedding_model.embed_query(text)

    def embed_texts(self, texts: list[str]):
        """Embed a list of texts."""
        return self.embedding_model.embed_documents(texts)
