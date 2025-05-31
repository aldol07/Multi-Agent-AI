from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import sqlite3
import redis
from pathlib import Path

class StorageBackend(ABC):
    @abstractmethod
    def store(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def retrieve(self, key: str) -> Any:
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        pass

class RedisBackend(StorageBackend):
    def __init__(self, url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(url)

    def store(self, key: str, value: Any) -> None:
        self.redis.set(key, json.dumps(value))

    def retrieve(self, key: str) -> Any:
        value = self.redis.get(key)
        return json.loads(value) if value else None

    def delete(self, key: str) -> None:
        self.redis.delete(key)

class SQLiteBackend(StorageBackend):
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

    def store(self, key: str, value: Any) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)",
                (key, json.dumps(value))
            )

    def retrieve(self, key: str) -> Any:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value FROM memory WHERE key = ?", (key,))
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None

    def delete(self, key: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memory WHERE key = ?", (key,))

class InMemoryBackend(StorageBackend):
    def __init__(self):
        self._storage: Dict[str, Any] = {}

    def store(self, key: str, value: Any) -> None:
        self._storage[key] = value

    def retrieve(self, key: str) -> Any:
        return self._storage.get(key)

    def delete(self, key: str) -> None:
        self._storage.pop(key, None)

class SharedMemory:
    def __init__(self, backend: StorageBackend = None):
        self.backend = backend or InMemoryBackend()

    def store_source_info(self, source_id: str, source_type: str, timestamp: Optional[datetime] = None) -> None:
        """Store source information including type and timestamp."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        data = {
            "source_type": source_type,
            "timestamp": timestamp.isoformat(),
        }
        self.backend.store(f"source:{source_id}", data)

    def store_extracted_values(self, source_id: str, values: Dict[str, Any]) -> None:
        """Store extracted values for a source."""
        self.backend.store(f"extracted:{source_id}", values)

    def store_thread_id(self, source_id: str, thread_id: str) -> None:
        """Store thread/conversation ID for a source."""
        self.backend.store(f"thread:{source_id}", thread_id)

    def store_processing_result(self, task_id: str, source: str, format_type: str, intent: str, extracted_data: Dict[str, Any]) -> None:
        """Store document processing results."""
        timestamp = datetime.utcnow()
        data = {
            "task_id": task_id,
            "source": source,
            "format_type": format_type,
            "intent": intent,
            "extracted_data": extracted_data,
            "timestamp": timestamp.isoformat()
        }
        self.backend.store(f"result:{task_id}", data)
        
        # Store in history
        history = self.get_recent_history(100) or []
        history.insert(0, data)
        self.backend.store("history", history[:100])  # Keep last 100 entries

    def get_recent_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent processing history."""
        history = self.backend.retrieve("history") or []
        return history[:limit]

    def get_source_info(self, source_id: str) -> Dict[str, Any]:
        """Retrieve source information."""
        return self.backend.retrieve(f"source:{source_id}")

    def get_extracted_values(self, source_id: str) -> Dict[str, Any]:
        """Retrieve extracted values."""
        return self.backend.retrieve(f"extracted:{source_id}")

    def get_thread_id(self, source_id: str) -> str:
        """Retrieve thread/conversation ID."""
        return self.backend.retrieve(f"thread:{source_id}")

    def delete_source_data(self, source_id: str) -> None:
        """Delete all data associated with a source."""
        self.backend.delete(f"source:{source_id}")
        self.backend.delete(f"extracted:{source_id}")
        self.backend.delete(f"thread:{source_id}")

# Factory function to create shared memory instance
def create_shared_memory(backend_type: str = "in_memory", **kwargs) -> SharedMemory:
    """
    Create a shared memory instance with the specified backend.
    
    Args:
        backend_type: One of "redis", "sqlite", or "in_memory"
        **kwargs: Additional arguments for the backend (e.g., redis_url, db_path)
    """
    if backend_type == "redis":
        return SharedMemory(RedisBackend(kwargs.get("redis_url", "redis://localhost:6379")))
    elif backend_type == "sqlite":
        return SharedMemory(SQLiteBackend(kwargs.get("db_path", "memory.db")))
    elif backend_type == "in_memory":
        return SharedMemory(InMemoryBackend())
    else:
        raise ValueError(f"Unknown backend type: {backend_type}") 