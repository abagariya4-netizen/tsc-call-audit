import os
import json
import hashlib
import threading
from pathlib import Path

CACHE_DIR = Path("cache")
LOCKS = {}
LOCKS_LOCK = threading.Lock()
USE_CACHE = True

def get_namespace_lock(namespace: str) -> threading.Lock:
    """Gets or creates a lock for the given namespace to prevent concurrent writes."""
    with LOCKS_LOCK:
        if namespace not in LOCKS:
            LOCKS[namespace] = threading.Lock()
        return LOCKS[namespace]

def audio_hash(audio_bytes: bytes) -> str:
    """Generates SHA256 hex digest of raw audio bytes."""
    return hashlib.sha256(audio_bytes).hexdigest()

def text_hash(*parts: str) -> str:
    """Generates SHA256 hex digest of concatenated text parts separated by a delimiter."""
    combined = "||".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

def _load_cache(namespace: str) -> dict:
    """Loads the entire JSON cache for the namespace. Internal use, must be called under lock."""
    CACHE_DIR.mkdir(exist_ok=True)
    file_path = CACHE_DIR / f"{namespace}.json"
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def cache_get(namespace: str, key: str) -> dict | None:
    """Retrieves a value from the cache. Returns None if USE_CACHE is False or key not found."""
    if not USE_CACHE:
        return None
    lock = get_namespace_lock(namespace)
    with lock:
        cache_data = _load_cache(namespace)
        return cache_data.get(key)

def cache_set(namespace: str, key: str, value: dict) -> None:
    """Saves a value in the cache in a thread-safe, atomic manner."""
    lock = get_namespace_lock(namespace)
    with lock:
        CACHE_DIR.mkdir(exist_ok=True)
        file_path = CACHE_DIR / f"{namespace}.json"
        
        # Read-modify-write
        cache_data = _load_cache(namespace)
        cache_data[key] = value
        
        # Write atomically using temp file
        temp_file_path = file_path.with_suffix(".tmp")
        try:
            with open(temp_file_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            os.replace(temp_file_path, file_path)
        except Exception as e:
            if temp_file_path.exists():
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass
            raise e
