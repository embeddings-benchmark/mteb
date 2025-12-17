"""
MongoDB Storage Implementation

Responsible for persisting event data to MongoDB
"""

import logging
import os
from typing import Any, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


class MongoDBStorage:
    """
    MongoDB storage class

    Supports reading connection string from environment variable MONGO_URI
    """

    # Default configuration
    DEFAULT_DATABASE = "event_logger"
    DEFAULT_COLLECTION = "events"

    def __init__(
        self,
        uri: Optional[str] = None,
        database: Optional[str] = None,
        collection: Optional[str] = None,
    ):
        """
        Initialize MongoDB connection

        Args:
            uri: MongoDB connection string, defaults to reading from environment variable MONGO_URI
            database: Database name, defaults to "event_logger"
            collection: Collection name, defaults to "events"
        """
        self._uri = uri or os.getenv("MONGO_URI")
        if not self._uri:
            raise ValueError(
                "MongoDB URI not configured. Please set environment variable MONGO_URI or pass uri parameter during initialization"
            )

        self._database_name = database or self.DEFAULT_DATABASE
        self._collection_name = collection or self.DEFAULT_COLLECTION

        # Lazy connection initialization
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._collection: Optional[Collection] = None

    def _ensure_connection(self) -> Collection:
        """
        Ensure database connection is established

        Returns:
            MongoDB Collection object
        """
        if self._collection is None:
            self._client = MongoClient(self._uri)
            self._db = self._client[self._database_name]
            self._collection = self._db[self._collection_name]
            logger.info(
                f"Connected to MongoDB: {self._database_name}.{self._collection_name}"
            )
        return self._collection

    def insert(self, event_data: dict[str, Any]) -> bool:
        """
        Insert a single event record

        Args:
            event_data: Event data dictionary

        Returns:
            Whether insertion was successful
        """
        try:
            collection = self._ensure_connection()
            result = collection.insert_one(event_data)
            logger.debug(f"Event recorded: {result.inserted_id}")
            return True
        except PyMongoError as e:
            logger.error(f"MongoDB write failed: {e}")
            return False

    def insert_many(self, events: list[dict[str, Any]]) -> bool:
        """
        Batch insert event records

        Args:
            events: List of event data

        Returns:
            Whether insertion was successful
        """
        if not events:
            return True

        try:
            collection = self._ensure_connection()
            result = collection.insert_many(events)
            logger.debug(f"Batch inserted {len(result.inserted_ids)} events")
            return True
        except PyMongoError as e:
            logger.error(f"MongoDB batch write failed: {e}")
            return False

    def close(self):
        """Close database connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._collection = None
            logger.info("MongoDB connection closed")
