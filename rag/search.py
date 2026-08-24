"""
Azure AI Search retrieval for the Computing History knowledge base.

This module:
1. Connects to a Microsoft Foundry project using DefaultAzureCredential.
2. Creates embeddings using a deployed embedding model.
3. Creates/validates an Azure AI Search vector index.
4. Uploads documents.
5. Performs hybrid/vector/keyword search.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Literal

from azure.ai.projects import AIProjectClient
from azure.identity import (
    ClientSecretCredential,
    get_bearer_token_provider,
)

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from dotenv import load_dotenv

from openai import OpenAI


SearchMode = Literal["hybrid", "vector", "keyword"]


@dataclass(frozen=True)
class RetrievedChunk:
    """A knowledge-base chunk plus metadata."""

    content: str
    title: str
    source: str
    chunk_id: str
    score: float | None


class ComputingHistorySearch:
    """Creates, populates, and queries an Azure AI Search vector index."""

    def __init__(self) -> None:

        # ---------------------------------------------------------
        # Load .env from the project directory
        # ---------------------------------------------------------

        load_dotenv()

        # ---------------------------------------------------------
        # Environment variables
        # ---------------------------------------------------------

        self.project_endpoint = (
            os.getenv("AZURE_AI_PROJECT_ENDPOINT")
            or os.getenv("AGENT_ENDPOINT")
        )

        # IMPORTANT:
        # This is the Azure AI resource endpoint.
        #
        # Example:
        # https://24wh5a0501-9855-resource.services.ai.azure.com
        #
        # It is different from the project endpoint.
        self.resource_endpoint = os.getenv(
            "AZURE_AI_RESOURCE_ENDPOINT"
        )

        self.search_endpoint = os.getenv(
            "AZURE_AI_SEARCH_ENDPOINT"
        )

        self.index_name = os.getenv(
            "AZURE_AI_SEARCH_INDEX_NAME"
        )

        # This MUST be the DEPLOYMENT NAME of your embedding model.
        #
        # For example:
        # text-embedding-3-small
        self.embedding_deployment = os.getenv(
            "AZURE_AI_EMBEDDING_DEPLOYMENT_NAME"
        )

        # ---------------------------------------------------------
        # Validate required environment variables
        # ---------------------------------------------------------

        missing = []

        if not self.project_endpoint:
            missing.append(
                "AZURE_AI_PROJECT_ENDPOINT or AGENT_ENDPOINT"
            )

        if not self.resource_endpoint:
            missing.append(
                "AZURE_AI_RESOURCE_ENDPOINT"
            )

        if not self.search_endpoint:
            missing.append(
                "AZURE_AI_SEARCH_ENDPOINT"
            )

        if not self.index_name:
            missing.append(
                "AZURE_AI_SEARCH_INDEX_NAME"
            )

        if not self.embedding_deployment:
            missing.append(
                "AZURE_AI_EMBEDDING_DEPLOYMENT_NAME"
            )

        if missing:
            raise ValueError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
            )

        # ---------------------------------------------------------
        # Print safe configuration information
        # ---------------------------------------------------------

        print("\n=== RAG Configuration ===")

        print(
            f"Foundry Project Endpoint : "
            f"{self.project_endpoint}"
        )

        print(
            f"Azure AI Resource Endpoint: "
            f"{self.resource_endpoint}"
        )

        print(
            f"Azure Search Endpoint    : "
            f"{self.search_endpoint}"
        )

        print(
            f"Search Index             : "
            f"{self.index_name}"
        )

        print(
            f"Embedding Deployment     : "
            f"{self.embedding_deployment}"
        )

        print("==========================\n")

        # ---------------------------------------------------------
        # Azure authentication
        # ---------------------------------------------------------

        credential = ClientSecretCredential(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET"),
)

        # ---------------------------------------------------------
        # Microsoft Foundry project
        #
        # This client remains available for project/agent
        # functionality.
        # ---------------------------------------------------------

        self._project_client = AIProjectClient(
            endpoint=self.project_endpoint.rstrip("/"),
            credential=credential,
        )

        # ---------------------------------------------------------
        # Azure OpenAI / Foundry resource client
        #
        # IMPORTANT:
        #
        # Embeddings are now created through the Azure AI resource
        # OpenAI-compatible /openai/v1/ endpoint.
        #
        # We are NOT using:
        #
        # self._project_client.get_openai_client()
        #
        # for embeddings.
        # ---------------------------------------------------------

        token_provider = get_bearer_token_provider(
            credential,
            "https://ai.azure.com/.default",
        )

        self._openai_client = OpenAI(
            base_url=(
                self.resource_endpoint.rstrip("/")
                + "/openai/v1/"
            ),
            api_key="unused",
            default_headers={
                "Authorization": f"Bearer {token_provider()}"
            },
        )

        # ---------------------------------------------------------
        # Azure AI Search
        #
        # Existing SearchIndexClient and SearchClient logic
        # remains unchanged.
        # ---------------------------------------------------------

        self._index_client = SearchIndexClient(
            endpoint=self.search_endpoint.rstrip("/"),
            credential=credential,
        )

        self._search_client = SearchClient(
            endpoint=self.search_endpoint.rstrip("/"),
            index_name=self.index_name,
            credential=credential,
        )

    # =============================================================
    # EMBEDDINGS
    # =============================================================

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Create embeddings using the deployed Azure AI model."""

        if not texts:
            return []

        print(
            f"Creating embeddings using deployment: "
            f"{self.embedding_deployment}"
        )

        try:
            response = self._openai_client.embeddings.create(
                model=self.embedding_deployment,
                input=texts,
            )

        except Exception as exc:
            print("\n❌ Embedding request failed.")

            print(
                f"Deployment: "
                f"{self.embedding_deployment}"
            )

            print(
                f"Resource endpoint: "
                f"{self.resource_endpoint}"
            )

            print(
                f"Embedding endpoint: "
                f"{self.resource_endpoint.rstrip('/')}/openai/v1/"
            )

            print(
                f"Error type: "
                f"{type(exc).__name__}"
            )

            print(
                f"Error: {exc}\n"
            )

            raise

        return [
            item.embedding
            for item in response.data
        ]

    # =============================================================
    # AZURE AI SEARCH INDEX
    # =============================================================

    def ensure_index(
        self,
        vector_dimensions: int,
    ) -> None:
        """Create the index or validate its vector dimensions."""

        try:
            existing = self._index_client.get_index(
                self.index_name
            )

        except Exception as exc:

            if exc.__class__.__name__ != "ResourceNotFoundError":
                raise

            existing = None

        # ---------------------------------------------------------
        # Existing index
        # ---------------------------------------------------------

        if existing is not None:

            vector_field = next(
                (
                    field
                    for field in existing.fields
                    if field.name == "content_vector"
                ),
                None,
            )

            if (
                vector_field is None
                or vector_field.vector_search_dimensions
                != vector_dimensions
            ):
                raise ValueError(
                    "The existing index has incompatible "
                    "vector dimensions.\n"
                    f"Expected: {vector_dimensions}\n"
                    f"Index: {self.index_name}\n\n"
                    "Use a new AZURE_AI_SEARCH_INDEX_NAME "
                    "or recreate the existing index."
                )

            print(
                f"✓ Azure AI Search index "
                f"'{self.index_name}' already exists."
            )

            return

        # ---------------------------------------------------------
        # Create new index
        # ---------------------------------------------------------

        fields = [

            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),

            SearchField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="en.lucene",
            ),

            SearchField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
            ),

            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
            ),

            SimpleField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                filterable=True,
            ),

            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single
                ),
                searchable=True,
                vector_search_dimensions=vector_dimensions,
                vector_search_profile_name="history-vector-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="history-hnsw"
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="history-vector-profile",
                    algorithm_configuration_name="history-hnsw",
                )
            ],
        )

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
        )

        self._index_client.create_index(index)

        print(
            f"✓ Created Azure AI Search index "
            f"'{self.index_name}'."
        )

    # =============================================================
    # UPLOAD DOCUMENTS
    # =============================================================

    def upload_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> None:
        """Upload documents to Azure AI Search."""

        if not documents:
            return

        print(
            f"Uploading {len(documents)} documents "
            "to Azure AI Search..."
        )

        result = self._search_client.upload_documents(
            documents=documents
        )

        failed = [
            item.key
            for item in result
            if not item.succeeded
        ]

        if failed:
            raise RuntimeError(
                "Azure AI Search rejected document IDs: "
                + ", ".join(failed)
            )

        print(
            "✓ Documents uploaded successfully."
        )

    # =============================================================
    # SEARCH
    # =============================================================

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: SearchMode = "hybrid",
    ) -> list[RetrievedChunk]:

        if not query or not query.strip():
            return []

        if mode not in {
            "hybrid",
            "vector",
            "keyword",
        }:
            raise ValueError(
                "mode must be 'hybrid', 'vector', or 'keyword'"
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1"
            )

        vector_queries = None
        search_text: str | None = query

        # ---------------------------------------------------------
        # Vector search
        # ---------------------------------------------------------

        if mode != "keyword":

            vector = self.embed([query])[0]

            vector_queries = [
                VectorizedQuery(
                    vector=vector,
                    k_nearest_neighbors=top_k,
                    fields="content_vector",
                )
            ]

        # ---------------------------------------------------------
        # Vector-only search
        # ---------------------------------------------------------

        if mode == "vector":
            search_text = None

        # ---------------------------------------------------------
        # Azure AI Search
        # ---------------------------------------------------------

        results = self._search_client.search(
            search_text=search_text,
            vector_queries=vector_queries,
            select=[
                "content",
                "title",
                "source",
                "chunk_id",
            ],
            top=top_k,
        )

        return [
            RetrievedChunk(
                content=result["content"],
                title=result["title"],
                source=result["source"],
                chunk_id=result["chunk_id"],
                score=result.get("@search.score"),
            )
            for result in results
        ]


# =============================================================
# COMMAND LINE SEARCH
# =============================================================

def _main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Query the Computing History RAG index."
        )
    )

    parser.add_argument(
        "query",
        help="Question or search phrase",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to return",
    )

    parser.add_argument(
        "--mode",
        choices=(
            "hybrid",
            "vector",
            "keyword",
        ),
        default="hybrid",
        help="Retrieval mode",
    )

    args = parser.parse_args()

    search = ComputingHistorySearch()

    results = search.search(
        args.query,
        top_k=args.top_k,
        mode=args.mode,
    )

    for position, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"[{position}] "
            f"{result.title} — "
            f"{result.source} "
            f"(chunk {result.chunk_id})"
        )

        print(result.content)
        print()


if __name__ == "__main__":
    _main()