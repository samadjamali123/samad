"""
Advanced Cache Management System for Plant Leaf Disease Detection

This module provides intelligent caching with multi-level strategies,
cache invalidation policies, distributed cache consistency, and performance monitoring.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from functools import wraps
from collections import OrderedDict
import redis.asyncio as redis
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration settings"""
    # Memory cache settings
    memory_max_size: int = 1000
    memory_ttl: int = 300  # 5 minutes
    
    # Redis cache settings
    redis_enabled: bool = True
    redis_ttl: int = 3600  # 1 hour
    redis_key_prefix: str = "plantdisease:"
    
    # Cache invalidation settings
    auto_invalidate: bool = True
    invalidate_on_update: bool = True
    background_cleanup: bool = True
    
    # Performance settings
    compression_enabled: bool = True
    serialization_method: str = "pickle"  # pickle, json
    async_backend: bool = True


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    errors: int = 0
    memory_size: int = 0
    redis_size: int = 0
    avg_hit_rate: float = 0.0
    avg_response_time: float = 0.0
    last_reset: float = field(default_factory=time.time)


class MemoryCache:
    """In-memory LRU cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.expiry_times = {}
        self.access_times = {}
        self.stats = CacheStats()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.expiry_times:
            return True
        
        return time.time() > self.expiry_times[key]
    
    def _evict_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, expiry_time in self.expiry_times.items()
            if current_time > expiry_time
        ]
        
        for key in expired_keys:
            self.delete(key)
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self.delete(oldest_key)
            self.stats.evictions += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            # Check if expired
            if self._is_expired(key):
                self.delete(key)
                self.stats.misses += 1
                return None
            
            # Move to end (most recently used)
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                self.access_times[key] = time.time()
                self.stats.hits += 1
                return value
            
            self.stats.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Memory cache get error: {e}")
            self.stats.errors += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        try:
            # Evict expired entries first
            self._evict_expired()
            
            # Set TTL
            if ttl is None:
                ttl = self.default_ttl
            
            expiry_time = time.time() + ttl
            
            # Evict LRU if necessary
            if key not in self.cache and len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Store value
            if key in self.cache:
                self.cache.pop(key)
            
            self.cache[key] = value
            self.expiry_times[key] = expiry_time
            self.access_times[key] = time.time()
            self.stats.sets += 1
            
        except Exception as e:
            logger.error(f"Memory cache set error: {e}")
            self.stats.errors += 1
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            deleted = False
            
            if key in self.cache:
                del self.cache[key]
                deleted = True
            
            if key in self.expiry_times:
                del self.expiry_times[key]
                deleted = True
            
            if key in self.access_times:
                del self.access_times[key]
                deleted = True
            
            if deleted:
                self.stats.deletes += 1
            
            return deleted
            
        except Exception as e:
            logger.error(f"Memory cache delete error: {e}")
            self.stats.errors += 1
            return False
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.expiry_times.clear()
        self.access_times.clear()
        self.stats.deletes += len(self.cache)
    
    def size(self) -> int:
        """Get current cache size"""
        self._evict_expired()
        return len(self.cache)
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests == 0:
            return 0.0
        return (self.stats.hits / total_requests) * 100
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "type": "memory",
            "size": self.size(),
            "max_size": self.max_size,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "evictions": self.stats.evictions,
            "errors": self.stats.errors,
            "hit_rate": self.hit_rate(),
            "last_reset": self.stats.last_reset
        }


class RedisCache:
    """Redis-based distributed cache with advanced features"""
    
    def __init__(self, redis_client: redis.Redis, key_prefix: str = "cache:"):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.stats = CacheStats()
        self.compression_enabled = True
    
    def _make_key(self, key: str) -> str:
        """Create full Redis key"""
        return f"{self.key_prefix}{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage"""
        try:
            if isinstance(value, (str, int, float, bool)):
                return str(value).encode('utf-8')
            elif isinstance(value, (dict, list)):
                return json.dumps(value, default=str).encode('utf-8')
            elif isinstance(value, np.ndarray):
                return pickle.dumps(value)
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize value from Redis storage"""
        try:
            # Try JSON first
            try:
                return json.loads(value.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            
            # Try pickle
            return pickle.loads(value)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return value
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            redis_key = self._make_key(key)
            value = await self.redis.get(redis_key)
            
            if value is None:
                self.stats.misses += 1
                return None
            
            self.stats.hits += 1
            return self._deserialize_value(value)
            
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")
            self.stats.errors += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in Redis cache"""
        try:
            redis_key = self._make_key(key)
            serialized_value = self._serialize_value(value)
            
            await self.redis.setex(redis_key, ttl, serialized_value)
            self.stats.sets += 1
            
        except Exception as e:
            logger.error(f"Redis cache set error: {e}")
            self.stats.errors += 1
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        try:
            redis_key = self._make_key(key)
            result = await self.redis.delete(redis_key)
            
            if result > 0:
                self.stats.deletes += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")
            self.stats.errors += 1
            return False
    
    async def clear(self, pattern: str = "*"):
        """Clear keys matching pattern"""
        try:
            search_pattern = self._make_key(pattern)
            keys = await self.redis.keys(search_pattern)
            
            if keys:
                deleted_count = await self.redis.delete(*keys)
                self.stats.deletes += deleted_count
                
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            self.stats.errors += 1
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            redis_key = self._make_key(key)
            return bool(await self.redis.exists(redis_key))
        except Exception as e:
            logger.error(f"Redis cache exists error: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get remaining TTL for key"""
        try:
            redis_key = self._make_key(key)
            return await self.redis.ttl(redis_key)
        except Exception as e:
            logger.error(f"Redis cache TTL error: {e}")
            return -1
    
    async def size(self) -> int:
        """Get approximate cache size"""
        try:
            pattern = self._make_key("*")
            keys = await self.redis.keys(pattern)
            return len(keys)
        except Exception as e:
            logger.error(f"Redis cache size error: {e}")
            return 0
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests == 0:
            return 0.0
        return (self.stats.hits / total_requests) * 100
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "type": "redis",
            "size": await self.size(),
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "evictions": self.stats.evictions,
            "errors": self.stats.errors,
            "hit_rate": self.hit_rate(),
            "last_reset": self.stats.last_reset
        }


class CacheManager:
    """Advanced multi-level cache manager"""
    
    def __init__(self, config: CacheConfig, redis_client: Optional[redis.Redis] = None):
        self.config = config
        self.memory_cache = MemoryCache(
            max_size=config.memory_max_size,
            default_ttl=config.memory_ttl
        )
        self.redis_cache = None
        
        if config.redis_enabled and redis_client:
            self.redis_cache = RedisCache(redis_client, config.redis_key_prefix)
        
        self.stats = CacheStats()
        self.invalidation_callbacks: List[Callable] = []
        
        # Background cleanup task
        self._cleanup_task = None
        
        logger.info("Cache manager initialized")
    
    async def initialize(self):
        """Initialize cache manager components"""
        try:
            # Start background cleanup task
            if self.config.background_cleanup:
                self._cleanup_task = asyncio.create_task(self._background_cleanup())
            
            logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup cache manager resources"""
        try:
            # Stop background cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Clear caches
            self.memory_cache.clear()
            if self.redis_cache:
                await self.redis_cache.clear()
            
            logger.info("Cache manager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cache manager cleanup: {e}")
    
    def _make_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(hash(str(arg))))
        
        # Add keyword arguments (sorted for consistency)
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        # Create final key
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (tries memory first, then Redis)"""
        try:
            # Try memory cache first
            value = self.memory_cache.get(key)
            if value is not None:
                self.stats.hits += 1
                return value
            
            # Try Redis cache
            if self.redis_cache:
                value = await self.redis_cache.get(key)
                if value is not None:
                    # Store in memory cache for faster access
                    self.memory_cache.set(key, value, self.config.memory_ttl)
                    self.stats.hits += 1
                    return value
            
            self.stats.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.stats.errors += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache (stores in both memory and Redis)"""
        try:
            # Set memory TTL
            memory_ttl = ttl or self.config.memory_ttl
            
            # Set Redis TTL
            redis_ttl = ttl or self.config.redis_ttl
            
            # Store in memory cache
            self.memory_cache.set(key, value, memory_ttl)
            
            # Store in Redis cache
            if self.redis_cache:
                await self.redis_cache.set(key, value, redis_ttl)
            
            self.stats.sets += 1
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.stats.errors += 1
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        try:
            memory_deleted = self.memory_cache.delete(key)
            redis_deleted = False
            
            if self.redis_cache:
                redis_deleted = await self.redis_cache.delete(key)
            
            if memory_deleted or redis_deleted:
                self.stats.deletes += 1
                
                # Trigger invalidation callbacks
                for callback in self.invalidation_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(key, "delete")
                        else:
                            callback(key, "delete")
                    except Exception as e:
                        logger.error(f"Invalidation callback error: {e}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            self.stats.errors += 1
            return False
    
    async def clear(self, pattern: str = "*"):
        """Clear cache entries matching pattern"""
        try:
            # Clear memory cache (full clear only)
            if pattern == "*":
                self.memory_cache.clear()
            
            # Clear Redis cache
            if self.redis_cache:
                await self.redis_cache.clear(pattern)
            
            self.stats.deletes += 1
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self.stats.errors += 1
    
    async def get_or_set(self, key: str, factory_func: Callable, 
                        ttl: Optional[int] = None) -> Any:
        """Get value from cache or set using factory function"""
        try:
            # Try to get from cache first
            cached_value = await self.get(key)
            if cached_value is not None:
                return cached_value
            
            # Generate value using factory function
            if asyncio.iscoroutinefunction(factory_func):
                value = await factory_func()
            else:
                value = factory_func()
            
            # Cache the generated value
            await self.set(key, value, ttl)
            
            return value
            
        except Exception as e:
            logger.error(f"Cache get_or_set error: {e}")
            raise
    
    def register_invalidation_callback(self, callback: Callable):
        """Register callback for cache invalidation events"""
        self.invalidation_callbacks.append(callback)
    
    async def _background_cleanup(self):
        """Background task for cache cleanup"""
        while True:
            try:
                # Sleep for cleanup interval
                await asyncio.sleep(300)  # 5 minutes
                
                # Trigger memory cleanup
                self.memory_cache._evict_expired()
                
                logger.debug("Background cache cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    def hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests == 0:
            return 0.0
        return (self.stats.hits / total_requests) * 100
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        redis_stats = {}
        
        if self.redis_cache:
            redis_stats = await self.redis_cache.get_stats()
        
        return {
            "overall": {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "sets": self.stats.sets,
                "deletes": self.stats.deletes,
                "errors": self.stats.errors,
                "hit_rate": self.hit_rate(),
                "last_reset": self.stats.last_reset
            },
            "memory": memory_stats,
            "redis": redis_stats,
            "config": {
                "memory_max_size": self.config.memory_max_size,
                "memory_ttl": self.config.memory_ttl,
                "redis_enabled": self.config.redis_enabled,
                "redis_ttl": self.config.redis_ttl,
                "compression_enabled": self.config.compression_enabled
            }
        }


# Decorator for automatic caching
def cached(prefix: str = "", ttl: int = 300, cache_manager: Optional[CacheManager] = None):
    """Decorator for automatic function result caching"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get cache manager
            cm = cache_manager or get_cache_manager()
            if not cm:
                # No caching available, just call function
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_key = cm._make_cache_key(
                prefix or func.__name__, 
                *args, 
                **kwargs
            )
            
            # Try to get from cache
            cached_result = await cm.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cm.set(cache_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to handle differently
            import asyncio
            
            # Get cache manager
            cm = cache_manager or get_cache_manager()
            if not cm:
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = cm._make_cache_key(
                prefix or func.__name__, 
                *args, 
                **kwargs
            )
            
            # Check if we're in an async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, need to run sync
                    return asyncio.create_task(
                        asyncio.to_thread(sync_wrapper_with_cache, func, args, kwargs, cm, cache_key, ttl)
                    )
            except RuntimeError:
                pass
            
            # Not in async context, use sync approach
            return sync_wrapper_with_cache(func, args, kwargs, cm, cache_key, ttl)
        
        def sync_wrapper_with_cache(original_func, args, kwargs, cm, cache_key, cache_ttl):
            """Helper function for sync caching"""
            # Try to get from cache (sync version)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                cached_result = loop.run_until_complete(cm.get(cache_key))
                if cached_result is not None:
                    return cached_result
            finally:
                loop.close()
            
            # Execute function
            result = original_func(*args, **kwargs)
            
            # Cache result
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(cm.set(cache_key, result, cache_ttl))
            finally:
                loop.close()
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> Optional[CacheManager]:
    """Get global cache manager instance"""
    return _cache_manager


def set_cache_manager(cache_manager: CacheManager):
    """Set global cache manager instance"""
    global _cache_manager
    _cache_manager = cache_manager