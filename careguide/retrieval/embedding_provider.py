"""OpenAI-compatible embedding adapter for CareGuide retrieval."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence


DEFAULT_QUERY_INSTRUCTION = (
    "Retrieve sourced medical information, medicine label sections, department "
    "descriptions, and hospital visit guides relevant to the query."
)


class OpenAICompatibleEmbeddingFunction:
    """Generate document and query embeddings through an OpenAI-compatible API.

    The callable interface keeps this object compatible with Chroma, while
    ``embed_documents`` and ``embed_query`` make the asymmetric query instruction
    explicit to the CareGuide ingestion and retrieval code.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        dimensions: int,
        encoding_format: str = "float",
        provider_name: str = "openai-compatible",
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = 64,
        timeout: float = 30.0,
        max_retries: int = 2,
        client: Any = None,
        response_observer: Any = None,
    ):
        if not str(api_key).strip():
            raise ValueError("embedding api_key cannot be empty")
        if not str(base_url).strip():
            raise ValueError("embedding base_url cannot be empty")
        if not str(model).strip():
            raise ValueError("embedding model cannot be empty")
        if int(dimensions) <= 0:
            raise ValueError("embedding dimensions must be positive")
        if int(batch_size) <= 0:
            raise ValueError("embedding batch_size must be positive")
        if encoding_format != "float":
            raise ValueError("CareGuide requires float embeddings")

        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )

        self._client = client
        self.response_observer = response_observer
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model).strip()
        self.dimensions = int(dimensions)
        self.encoding_format = encoding_format
        self.provider_name = str(provider_name).strip() or "openai-compatible"
        self.query_instruction = str(query_instruction).strip()
        self.batch_size = int(batch_size)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Chroma-compatible document embedding entry point."""
        # Chroma 0.5 HTTP serializes each vector with .tolist(). Keep the
        # explicit retrieval methods returning ordinary lists.
        import numpy as np
        return [np.asarray(vector, dtype=np.float32) for vector in self.embed_documents(input)]

    def embed_documents(self, texts: Sequence[str], *, timeout: float | None = None) -> List[List[float]]:
        return self._embed([str(text) for text in texts], timeout=timeout)

    def embed_query(self, query: str, *, timeout: float | None = None) -> List[float]:
        return self.embed_queries([query], timeout=timeout)[0]

    def embed_queries(self, queries: Sequence[str], *, timeout: float | None = None) -> List[List[float]]:
        """Batch query embeddings while preserving the query-side instruction."""
        query_texts: List[str] = []
        for query in queries:
            query_text = str(query).strip()
            if not query_text:
                raise ValueError("embedding query cannot be empty")
            if self.query_instruction:
                query_text = f"Instruct: {self.query_instruction}\nQuery: {query_text}"
            query_texts.append(query_text)
        return self._embed(query_texts, timeout=timeout)

    def _embed(self, texts: List[str], *, timeout: float | None = None) -> List[List[float]]:
        if not texts:
            return []

        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                encoding_format=self.encoding_format,
                **({'timeout': timeout} if timeout is not None else {}),
            )
            if self.response_observer is not None:
                self.response_observer(response)
            data = list(getattr(response, "data", None) or [])
            if len(data) != len(batch):
                raise RuntimeError(
                    "embedding API returned an unexpected number of vectors: "
                    f"expected={len(batch)} actual={len(data)}"
                )

            indexed = list(enumerate(data))
            indexed.sort(key=lambda pair: int(getattr(pair[1], "index", pair[0])))
            for _, item in indexed:
                raw_vector = getattr(item, "embedding", None)
                if raw_vector is None:
                    raise RuntimeError("embedding API response is missing embedding data")
                vector = [float(value) for value in raw_vector]
                self._validate_vector(vector)
                vectors.append(vector)
        return vectors

    def _validate_vector(self, vector: List[float]) -> None:
        if len(vector) != self.dimensions:
            raise RuntimeError(
                "embedding API returned the wrong vector dimension: "
                f"expected={self.dimensions} actual={len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError("embedding API returned a non-finite vector")

    def name(self) -> str:
        normalized_model = self.model.casefold().replace("/", "-")
        return f"careguide-{self.provider_name.casefold()}-{normalized_model}"

    def close(self):
        self._client.close()

    def metadata(self) -> Dict[str, Any]:
        return {
            "embedding_provider": self.provider_name,
            "embedding_model": self.model,
            "embedding_dimensions": self.dimensions,
            "embedding_format": self.encoding_format,
        }
