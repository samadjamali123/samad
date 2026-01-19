"""
Performance Optimization Utilities for Plant Leaf Disease Detection System

This module provides advanced performance optimization and monitoring capabilities
for production scalability, including database optimization, caching strategies,
auto-scaling policies, and resource management.
"""

import asyncio
import gc
import logging
import psutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Callable
from functools import wraps, lru_cache
import numpy as np
import torch
import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.pool import QueuePool
import uvloop

# Configure logging
logger = logging.getLogger(__name__)

# Set event loop policy for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_mb: float = 0.0
    request_count: int = 0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    db_connections: int = 0
    active_threads: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResourceLimits:
    """Resource utilization limits and thresholds"""
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 85.0
    max_memory_mb: float = 4096.0
    max_disk_usage_percent: float = 90.0
    max_network_io_mb: float = 1000.0
    max_request_count: int = 1000
    max_avg_response_time: float = 2.0
    max_error_rate: float = 5.0
    min_cache_hit_rate: float = 70.0
    max_db_connections: int = 50
    max_active_threads: int = 100


class PerformanceOptimizer:
    """Advanced performance optimization and monitoring system"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.metrics = PerformanceMetrics()
        self.limits = ResourceLimits()
        self._monitoring_active = False
        self._monitoring_thread = None
        self._request_times = []
        self._error_count = 0
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._optimization_callbacks: List[Callable] = []
        
        # Thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="perf_opt"
        )
        
        # Connection pools
        self.db_pool = None
        self.redis_pool = None
        
        # Performance optimization caches
        self._model_cache = {}
        self._result_cache = {}
        self._image_cache = {}
        
        logger.info("Performance optimizer initialized")
    
    async def initialize(self):
        """Initialize performance optimization components"""
        try:
            # Initialize Redis connection pool for caching
            if self.settings and hasattr(self.settings, 'REDIS_URL'):
                self.redis_pool = redis.ConnectionPool.from_url(
                    self.settings.REDIS_URL,
                    max_connections=20,
                    retry_on_timeout=True
                )
            
            # Start performance monitoring
            await self.start_monitoring()
            
            # Setup garbage collection optimization
            gc.set_threshold(700, 10, 10)
            
            logger.info("Performance optimizer components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance optimizer: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup performance optimization resources"""
        try:
            # Stop monitoring
            await self.stop_monitoring()
            
            # Shutdown thread pool
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            # Close connection pools
            if self.redis_pool:
                await self.redis_pool.disconnect()
            
            # Clear caches
            self._model_cache.clear()
            self._result_cache.clear()
            self._image_cache.clear()
            
            logger.info("Performance optimizer cleaned up")
            
        except Exception as e:
            logger.error(f"Error during performance optimizer cleanup: {e}")
    
    def register_optimization_callback(self, callback: Callable):
        """Register callback for performance optimization events"""
        self._optimization_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start performance monitoring thread"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitor_performance,
                daemon=True
            )
            self._monitoring_thread.start()
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring thread"""
        if self._monitoring_active:
            self._monitoring_active = False
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5)
            logger.info("Performance monitoring stopped")
    
    def _monitor_performance(self):
        """Monitor system performance in background thread"""
        while self._monitoring_active:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check optimization triggers
                self._check_optimization_triggers()
                
                # Sleep for monitoring interval
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(10)
    
    def _update_system_metrics(self):
        """Update current system performance metrics"""
        try:
            # CPU and Memory
            self.metrics.cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            self.metrics.memory_percent = memory.percent
            self.metrics.memory_usage_mb = memory.used / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics.disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.metrics.network_io_mb = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)
            
            # Thread count
            self.metrics.active_threads = threading.active_count()
            
            # Request metrics
            if self._total_requests > 0:
                self.metrics.avg_response_time = np.mean(self._request_times[-100:]) if self._request_times else 0
                self.metrics.error_rate = (self._error_count / self._total_requests) * 100
                total_cache_ops = self._cache_hits + self._cache_misses
                self.metrics.cache_hit_rate = (self._cache_hits / total_cache_ops * 100) if total_cache_ops > 0 else 0
            
            # GPU utilization if available
            try:
                if torch.cuda.is_available():
                    self.metrics.gpu_utilization = torch.cuda.utilization()
            except:
                self.metrics.gpu_utilization = 0.0
            
            self.metrics.timestamp = time.time()
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _check_optimization_triggers(self):
        """Check if performance optimization should be triggered"""
        optimizations_triggered = []
        
        # Check CPU usage
        if self.metrics.cpu_percent > self.limits.max_cpu_percent:
            optimizations_triggered.append("high_cpu")
        
        # Check memory usage
        if self.metrics.memory_percent > self.limits.max_memory_percent:
            optimizations_triggered.append("high_memory")
        
        # Check response time
        if self.metrics.avg_response_time > self.limits.max_avg_response_time:
            optimizations_triggered.append("slow_response")
        
        # Check error rate
        if self.metrics.error_rate > self.limits.max_error_rate:
            optimizations_triggered.append("high_error_rate")
        
        # Check cache hit rate
        if self.metrics.cache_hit_rate < self.limits.min_cache_hit_rate:
            optimizations_triggered.append("low_cache_hit_rate")
        
        # Trigger optimization callbacks
        for trigger in optimizations_triggered:
            await self._trigger_optimization(trigger)
    
    async def _trigger_optimization(self, trigger: str):
        """Trigger performance optimization for specific issue"""
        logger.warning(f"Performance optimization triggered: {trigger}")
        
        # Execute optimization callbacks
        for callback in self._optimization_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(trigger, self.metrics)
                else:
                    callback(trigger, self.metrics)
            except Exception as e:
                logger.error(f"Optimization callback error: {e}")
        
        # Apply built-in optimizations
        if trigger == "high_memory":
            await self._optimize_memory_usage()
        elif trigger == "high_cpu":
            await self._optimize_cpu_usage()
        elif trigger == "slow_response":
            await self._optimize_response_times()
        elif trigger == "low_cache_hit_rate":
            await self._optimize_cache_performance()
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage"""
        try:
            # Clear caches if memory is high
            if self.metrics.memory_percent > 90:
                self._model_cache.clear()
                self._result_cache.clear()
                self._image_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Memory optimization completed")
            
        except Exception as e:
            logger.error(f"Memory optimization error: {e}")
    
    async def _optimize_cpu_usage(self):
        """Optimize CPU usage"""
        try:
            # Reduce thread pool size if CPU is high
            if self.metrics.cpu_percent > 85:
                self.thread_pool._max_workers = max(2, self.thread_pool._max_workers - 1)
            
            logger.info("CPU optimization completed")
            
        except Exception as e:
            logger.error(f"CPU optimization error: {e}")
    
    async def _optimize_response_times(self):
        """Optimize response times"""
        try:
            # Clear slow request tracking
            if len(self._request_times) > 1000:
                self._request_times = self._request_times[-500:]
            
            logger.info("Response time optimization completed")
            
        except Exception as e:
            logger.error(f"Response time optimization error: {e}")
    
    async def _optimize_cache_performance(self):
        """Optimize cache performance"""
        try:
            # Clear old cache entries
            current_time = time.time()
            
            # Clear model cache entries older than 1 hour
            self._model_cache = {
                k: v for k, v in self._model_cache.items()
                if current_time - v.get('timestamp', 0) < 3600
            }
            
            # Clear result cache entries older than 30 minutes
            self._result_cache = {
                k: v for k, v in self._result_cache.items()
                if current_time - v.get('timestamp', 0) < 1800
            }
            
            logger.info("Cache performance optimization completed")
            
        except Exception as e:
            logger.error(f"Cache optimization error: {e}")
    
    def record_request(self, response_time: float, is_error: bool = False):
        """Record request metrics for performance tracking"""
        self._total_requests += 1
        self._request_times.append(response_time)
        
        if is_error:
            self._error_count += 1
        
        # Keep only recent request times
        if len(self._request_times) > 1000:
            self._request_times = self._request_times[-500:]
    
    def record_cache_hit(self):
        """Record cache hit for performance tracking"""
        self._cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss for performance tracking"""
        self._cache_misses += 1
    
    @asynccontextmanager
    async def measure_performance(self, operation_name: str):
        """Context manager to measure operation performance"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.info(
                f"Performance: {operation_name} - "
                f"Duration: {duration:.3f}s, "
                f"Memory delta: {memory_delta:.2f}MB"
            )
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        if self.metrics.cpu_percent > self.limits.max_cpu_percent:
            recommendations.append(
                f"High CPU usage ({self.metrics.cpu_percent:.1f}%). "
                "Consider scaling horizontally or optimizing CPU-intensive operations."
            )
        
        if self.metrics.memory_percent > self.limits.max_memory_percent:
            recommendations.append(
                f"High memory usage ({self.metrics.memory_percent:.1f}%). "
                "Consider optimizing memory usage or increasing resources."
            )
        
        if self.metrics.avg_response_time > self.limits.max_avg_response_time:
            recommendations.append(
                f"Slow response times ({self.metrics.avg_response_time:.2f}s). "
                "Consider optimizing database queries or implementing caching."
            )
        
        if self.metrics.error_rate > self.limits.max_error_rate:
            recommendations.append(
                f"High error rate ({self.metrics.error_rate:.1f}%). "
                "Investigate error causes and improve error handling."
            )
        
        if self.metrics.cache_hit_rate < self.limits.min_cache_hit_rate:
            recommendations.append(
                f"Low cache hit rate ({self.metrics.cache_hit_rate:.1f}%). "
                "Consider optimizing cache strategies or increasing cache size."
            )
        
        return recommendations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "metrics": {
                "cpu_percent": self.metrics.cpu_percent,
                "memory_percent": self.metrics.memory_percent,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "gpu_utilization": self.metrics.gpu_utilization,
                "disk_usage_percent": self.metrics.disk_usage_percent,
                "network_io_mb": self.metrics.network_io_mb,
                "request_count": self._total_requests,
                "avg_response_time": self.metrics.avg_response_time,
                "error_rate": self.metrics.error_rate,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "active_threads": self.metrics.active_threads,
                "timestamp": self.metrics.timestamp
            },
            "limits": {
                "max_cpu_percent": self.limits.max_cpu_percent,
                "max_memory_percent": self.limits.max_memory_percent,
                "max_avg_response_time": self.limits.max_avg_response_time,
                "max_error_rate": self.limits.max_error_rate,
                "min_cache_hit_rate": self.limits.min_cache_hit_rate
            },
            "recommendations": self.get_optimization_recommendations(),
            "cache_stats": {
                "model_cache_size": len(self._model_cache),
                "result_cache_size": len(self._result_cache),
                "image_cache_size": len(self._image_cache),
                "total_cache_hits": self._cache_hits,
                "total_cache_misses": self._cache_misses
            }
        }


# Performance monitoring decorators
def monitor_performance(operation_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    logger.info(f"Performance: {name} took {duration:.3f}s")
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    logger.info(f"Performance: {name} took {duration:.3f}s")
            return sync_wrapper
    return decorator


def cache_result(ttl: int = 300, max_size: int = 100):
    """Decorator to cache function results with TTL"""
    def decorator(func):
        cache = {}
        cache_times = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check cache
            if cache_key in cache:
                if current_time - cache_times[cache_key] < ttl:
                    return cache[cache_key]
                else:
                    # Remove expired entry
                    del cache[cache_key]
                    del cache_times[cache_key]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if len(cache) >= max_size:
                # Remove oldest entry
                oldest_key = min(cache_times.keys(), key=cache_times.get)
                del cache[oldest_key]
                del cache_times[oldest_key]
            
            cache[cache_key] = result
            cache_times[cache_key] = current_time
            
            return result
        
        wrapper.cache_clear = cache.clear
        return wrapper
    return decorator


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> Optional[PerformanceOptimizer]:
    """Get global performance optimizer instance"""
    return _performance_optimizer


def set_performance_optimizer(optimizer: PerformanceOptimizer):
    """Set global performance optimizer instance"""
    global _performance_optimizer
    _performance_optimizer = optimizer


# Database connection pool optimization
def create_optimized_db_pool(database_url: str, **kwargs) -> QueuePool:
    """Create optimized database connection pool"""
    default_kwargs = {
        "poolclass": QueuePool,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 3600,
        "pool_pre_ping": True,
        "echo": False
    }
    default_kwargs.update(kwargs)
    
    from sqlalchemy import create_engine
    engine = create_engine(database_url, **default_kwargs)
    
    return engine.pool()


# Memory optimization utilities
class MemoryOptimizer:
    """Memory optimization utilities for large operations"""
    
    @staticmethod
    def optimize_numpy_operations():
        """Optimize NumPy operations for better memory usage"""
        import os
        # Set NumPy threads for better performance
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        os.environ['NUMEXPR_NUM_THREADS'] = '4'
    
    @staticmethod
    @asynccontextmanager
    async def memory_efficient_image_processing():
        """Context manager for memory-efficient image processing"""
        try:
            # Enable memory optimization
            MemoryOptimizer.optimize_numpy_operations()
            
            # Clear any existing caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            yield
            
        finally:
            # Cleanup memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @staticmethod
    def batch_process_memory_efficient(items: List[Any], 
                                     process_func: Callable,
                                     batch_size: int = 10) -> List[Any]:
        """Process items in memory-efficient batches"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            try:
                batch_results = process_func(batch)
                results.extend(batch_results)
                
                # Cleanup memory between batches
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {e}")
                continue
        
        return results