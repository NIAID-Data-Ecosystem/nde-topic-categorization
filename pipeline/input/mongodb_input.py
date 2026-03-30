# MongoDB input implementation
import logging
from typing import Any, Dict, Iterator, List, Optional

from ..core import TextItem
from .base import DataInput

logger = logging.getLogger(__name__)


class MongoDBInput(DataInput):
    def __init__(
        self,
        uri: str,
        database: str,
        collection: str,
        text_columns: Optional[List[str]] = None,
        metadata_mapping: Optional[Dict[str, str]] = None,
        id_field: str = "_id",
        text_separator: str = " ",
        query: Optional[Dict[str, Any]] = None,
        limit: int = 0,
    ):
        """
        Initialize MongoDB input handler.

        Args:
            uri: MongoDB connection URI (e.g. "mongodb://su11:27017")
            database: Name of the MongoDB database
            collection: Name of the MongoDB collection
            text_columns: Document fields to combine into the text passed to the LLM
            metadata_mapping: Dict mapping document fields to metadata keys
                e.g. {"name": "title", "url": "source_url"}
            id_field: Document field to use as item ID (default: "_id")
            text_separator: String used when joining multiple text fields
            query: Optional MongoDB filter query (default: all documents)
            limit: Maximum number of documents to read; 0 means no limit
        """
        self.uri = uri
        self.database = database
        self.collection = collection
        self.text_columns = text_columns or ["description"]
        self.metadata_mapping = metadata_mapping or {}
        self.id_field = id_field
        self.text_separator = text_separator
        self.query = query or {}
        self.limit = limit

    # ------------------------------------------------------------------
    # DataInput interface
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Check that the MongoDB connection and collection are reachable."""
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure, OperationFailure

            client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            db = client[self.database]
            collection_names = db.list_collection_names()
            client.close()

            if self.collection not in collection_names:
                logger.error(
                    f"Collection '{self.collection}' not found in database '{self.database}'. "
                    f"Available collections: {collection_names}"
                )
                return False

            return True
        except Exception as e:
            logger.error(f"MongoDB validation failed: {e}")
            return False

    def read(self) -> Iterator[TextItem]:
        """Connect to MongoDB, stream documents and yield TextItems."""
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "pymongo is required for MongoDBInput. "
                "Install it with: pip install pymongo"
            )

        client = MongoClient(self.uri)
        try:
            col = client[self.database][self.collection]
            cursor = col.find(self.query)
            if self.limit:
                cursor = cursor.limit(self.limit)

            for doc in cursor:
                yield TextItem(
                    id=self._extract_id(doc),
                    text=self._combine_text(doc),
                    metadata=self._extract_metadata(doc),
                )
        finally:
            client.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_id(self, doc: Dict[str, Any]) -> str:
        """Return the document ID as a string."""
        value = doc.get(self.id_field)
        if value is None:
            return ""
        return str(value)

    def _combine_text(self, doc: Dict[str, Any]) -> str:
        """Combine configured text fields into a single string."""
        parts = [
            str(doc[field]).strip()
            for field in self.text_columns
            if field in doc and doc[field] is not None
        ]
        return self.text_separator.join(parts) if parts else "No Description"

    def _extract_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata fields according to the mapping."""
        metadata: Dict[str, Any] = {}
        for doc_field, meta_key in self.metadata_mapping.items():
            value = doc.get(doc_field)
            if value is not None:
                metadata[meta_key] = str(value)
        return metadata
