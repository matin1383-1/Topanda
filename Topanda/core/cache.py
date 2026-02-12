
import numpy as np
from typing import Dict, List

class DistanceCache:
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []
    
    def get(self, key: str) :
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    def set(self, key: str, matrix: np.ndarray) -> None:
        if self.max_size == 0:
            return
        # If key already exists, remove from access order
        if key in self._cache:
            self._access_order.remove(key)
        # If at capacity, evict oldest (first in list)
        elif len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        # Add new entry
        self._cache[key] = matrix
        self._access_order.append(key)
    
    def invalidate(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)
    
    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()
    
    @property
    def size(self) -> int:
        return len(self._cache)
    
