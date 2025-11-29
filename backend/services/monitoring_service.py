"""
Monitoring and Analytics Service

This module provides comprehensive monitoring, analytics, and performance tracking
for the Plant Leaf Disease Detection system. It includes model performance monitoring,
user behavior analytics, system health tracking, and business intelligence.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import psutil
import os
import threading
from pathlib import Path

import aioredis
from pydantic import BaseModel, Field, validator
import prometheus_client
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, Summary, push_to_gateway

from backend.services.config import Settings
from backend.models.analysis import AIModelType, AIModelResult

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for AI model performance"""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_processing_time_ms: float = 0.0
    confidence_sum: float = 0.0
    confidence_count: int = 0
    last_request_time: Optional[datetime] = None
    error_rate: float = 0.0
    uptime_percentage: float = 100.0


@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    active_connections: int = 0
    uptime_seconds: int = 0
    error_count: int = 0
    request_rate_per_minute: float = 0.0


@dataclass
class BusinessMetrics:
    """Business intelligence metrics"""
    total_analyses: int = 0
    unique_users: int = 0
    disease_detections: int = 0
    healthy_detections: int = 0
    top_diseases: Dict[str, int] = field(default_factory=dict)
    top_plants: Dict[str, int] = field(default_factory=dict)
    avg_confidence_score: float = 0.0
    user_satisfaction_score: float = 0.0


@dataclass
class AlertMetrics:
    """System alert and notification metrics"""
    critical_alerts: int = 0
    warning_alerts: int = 0
    info_alerts: int = 0
    total_alerts: int = 0
    last_alert_time: Optional[datetime] = None
    active_alerts: List[str] = field(default_factory=list)


class PerformanceThresholds(BaseModel):
    """Performance alert thresholds"""
    cpu_warning_threshold: float = Field(70.0, description="CPU usage warning threshold %")
    cpu_critical_threshold: float = Field(85.0, description="CPU usage critical threshold %")
    memory_warning_threshold: float = Field(75.0, description="Memory usage warning threshold %")
    memory_critical_threshold: float = Field(90.0, description="Memory usage critical threshold %")
    error_rate_warning_threshold: float = Field(0.05, description="Error rate warning threshold")
    error_rate_critical_threshold: float = Field(0.10, description="Error rate critical threshold")
    response_time_warning_threshold: float = Field(2000.0, description="Response time warning threshold ms")
    response_time_critical_threshold: float = Field(5000.0, description="Response time critical threshold ms")
    confidence_warning_threshold: float = Field(0.6, description="Average confidence warning threshold")
    confidence_critical_threshold: float = Field(0.4, description="Average confidence critical threshold")

    @validator('response_time_warning_threshold', 'response_time_critical_threshold')
    def validate_response_times(cls, v):
        if v['response_time_warning_threshold'] >= v['response_time_critical_threshold']:
            raise ValueError("Warning threshold must be less than critical threshold")
        return v


class MonitoringService:
    """Comprehensive monitoring and analytics service"""
    
    def __init__(self, settings: Settings):
        """Initialize monitoring service with settings"""
        self.settings = settings
        self.start_time = time.time()
        
        # Initialize metrics storage
        self.model_metrics: Dict[str, ModelMetrics] = {
            model_type.value: ModelMetrics(
                model_name=model_type.value
            ) for model_type in AIModelType
        }
        
        self.system_metrics = SystemMetrics()
        self.business_metrics = BusinessMetrics()
        self.alert_metrics = AlertMetrics()
        
        # Performance tracking
        self.performance_thresholds = PerformanceThresholds()
        
        # Redis for metrics storage (if available)
        self.redis_client = None
        self.metrics_key_prefix = "plant_disease_detector:"
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.setup_prometheus_metrics()
        
        # Alert system
        self.alert_subscribers = []
        self.alert_history = deque(maxlen=1000)
        
        logger.info("Monitoring service initialized")

    async def initialize(self):
        """Initialize monitoring service components"""
        logger.info("Initializing monitoring service...")
        
        # Initialize Redis if configured
        if hasattr(self.settings, 'REDIS_URL') and self.settings.REDIS_URL:
            try:
                self.redis_client = aioredis.from_url(
                    self.settings.REDIS_URL,
                    decode_responses=True
                )
                logger.info("Redis client initialized for metrics storage")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis client: {str(e)}")
                self.redis_client = None
        
        # Start background monitoring tasks
        await self._start_monitoring_tasks()
        
        logger.info("Monitoring service initialization complete")

    async def cleanup(self):
        """Clean up monitoring service resources"""
        logger.info("Cleaning up monitoring service...")
        
        # Cancel background tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Monitoring service cleanup complete")

    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics collection"""
        try:
            # Model performance metrics
            for model_type in AIModelType:
                model_name = model_type.value.replace('_', '-').lower()
                
                # Request counters
                self.registry.register(
                    f"plant_detector_{model_name}_requests_total",
                    Counter(f"Total requests for {model_name} model", ["model", "service"])
                )
                
                self.registry.register(
                    f"plant_detector_{model_name}_requests_success",
                    Counter(f"Successful requests for {model_name} model", ["model", "status"])
                )
                
                self.registry.register(
                    f"plant_detector_{model_name}_requests_failed",
                    Counter(f"Failed requests for {model_name} model", ["model", "status"])
                )
                
                # Performance metrics
                self.registry.register(
                    f"plant_detector_{model_name}_processing_time",
                    Histogram(f"Processing time for {model_name} model", ["model"], ["milliseconds"])
                )
                
                self.registry.register(
                    f"plant_detector_{model_name}_confidence_score",
                    Histogram(f"Confidence scores for {model_name} model", ["model"])
                )
                
                # Gauge for current status
                self.registry.register(
                    f"plant_detector_{model_name}_error_rate",
                    Gauge(f"Error rate for {model_name} model", ["model"])
                )
                
                self.registry.register(
                    f"plant_detector_{model_name}_uptime",
                    Gauge(f"Uptime for {model_name} model", ["model"])
                )
            
            # System metrics
            self.registry.register(
                "plant_detector_system_cpu_usage",
                Gauge("System CPU usage", ["system"])
            )
            
            self.registry.register(
                "plant_detector_system_memory_usage",
                Gauge("System memory usage", ["system"])
            )
            
            self.registry.register(
                "plant_detector_system_disk_usage",
                Gauge("System disk usage", ["system"])
            )
            
            self.registry.register(
                "plant_detector_system_active_connections",
                Gauge("Active connections", ["system"])
            )
            
            # Business metrics
            self.registry.register(
                "plant_detector_analyses_total",
                Counter("Total analyses", ["business"])
            )
            
            self.registry.register(
                "plant_detector_unique_users_total",
                Gauge("Unique users", ["business"])
            )
            
            self.registry.register(
                "plant_detector_disease_detections_total",
                Counter("Disease detections", ["business"])
            )
            
            self.registry.register(
                "plant_detector_healthy_detections_total",
                Counter("Healthy detections", ["business"])
            )
            
            self.registry.register(
                "plant_detector_avg_confidence",
                Gauge("Average confidence score", ["business"])
            )
            
            self.registry.register(
                "plant_detector_alerts_total",
                Counter("Total alerts", ["system"])
            )
            
            logger.info("Prometheus metrics registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Prometheus metrics: {str(e)}")

    async def record_model_prediction(self, model_type: AIModelType, result: AIModelResult):
        """Record metrics for a model prediction"""
        if not self.model_metrics.get(model_type.value):
            return
        
        metrics = self.model_metrics[model_type.value]
        current_time = datetime.now()
        
        # Update basic metrics
        metrics.total_requests += 1
        
        if result.success:
            metrics.successful_requests += 1
            if result.disease_identification:
                metrics.confidence_sum += result.disease_identification.confidence
                metrics.confidence_count += 1
        else:
            metrics.failed_requests += 1
        
        metrics.last_request_time = current_time
        
        # Update error rate
        if metrics.total_requests > 0:
            metrics.error_rate = metrics.failed_requests / metrics.total_requests
        else:
            metrics.error_rate = 0.0
        
        # Update uptime
        if self.start_time:
            metrics.uptime_percentage = ((current_time.timestamp() - self.start_time) / 3600) * 100
        
        # Update Prometheus metrics
        try:
            # Increment request counter
            model_name = model_type.value.replace('_', '-').lower()
            
            if result.success:
                self.registry.get(f"plant_detector_{model_name}_requests_success").inc()
                if result.disease_identification:
                    self.registry.get(f"plant_detector_{model_name}_confidence_score").observe(
                        result.disease_identification.confidence
                    )
            else:
                self.registry.get(f"plant_detector_{model_name}_requests_failed").inc()
            else:
                self.registry.get(f"plant_detector_{model_name}_requests_total").inc()
            
            # Update processing time
            if result.processing_time_ms:
                self.registry.get(f"plant_detector_{model_name}_processing_time").observe(
                    result.processing_time_ms
                )
            
            # Update error rate gauge
            self.registry.get(f"plant_detector_{model_name}_error_rate").set(metrics.error_rate)
            
            # Update uptime gauge
            self.registry.get(f"plant_detector_{model_name}_uptime").set(metrics.uptime_percentage)
            
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {str(e)}")
        
        # Store in Redis if available
        await self._store_metrics_in_redis(model_type.value, metrics)

    async def record_system_metrics(self):
        """Record system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_metrics.cpu_usage_percent = cpu_percent
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_metrics.memory_usage_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.system_metrics.disk_usage_percent = disk.percent
            
            # Network I/O (basic)
            network = psutil.net_io_counters()
            self.system_metrics.network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Active connections (simplified)
            self.system_metrics.active_connections = len(psutil.net_connections())
            
            # Uptime
            self.system_metrics.uptime_seconds = int(time.time() - self.start_time)
            
            # Update Prometheus metrics
            self.registry.get("plant_detector_system_cpu_usage").set(cpu_percent)
            self.registry.get("plant_detector_system_memory_usage").set(memory.percent)
            self.registry.get("plant_detector_system_disk_usage").set(disk.percent)
            self.registry.get("plant_detector_system_active_connections").set(self.system_metrics.active_connections)
            
        except Exception as e:
            logger.error(f"Failed to record system metrics: {str(e)}")

    async def record_business_metrics(self, user_id: str = None, disease_detected: bool = None, 
                               plant_name: str = None, confidence: float = None):
        """Record business intelligence metrics"""
        current_time = datetime.now()
        
        # Update analysis count
        self.business_metrics.total_analyses += 1
        
        # Update unique users (simplified tracking)
        if user_id and user_id not in getattr(self, '_tracked_users', set()):
            self.business_metrics.unique_users += 1
            if not hasattr(self, '_tracked_users'):
                self._tracked_users = set()
            self._tracked_users.add(user_id)
        
        # Update disease/healthy counts
        if disease_detected is not None:
            if disease_detected:
                self.business_metrics.disease_detections += 1
            else:
                self.business_metrics.healthy_detections += 1
        
        # Update top diseases and plants
        if disease_detected and disease_detected != "healthy":
            self.business_metrics.top_diseases[disease_detected] = \
                self.business_metrics.top_diseases.get(disease_detected, 0) + 1
        
        if plant_name:
            self.business_metrics.top_plants[plant_name] = \
                self.business_metrics.top_plants.get(plant_name, 0) + 1
        
        # Update average confidence
        if confidence is not None:
            if self.business_metrics.confidence_count > 0:
                # Calculate rolling average
                total_confidence = self.business_metrics.confidence_sum + confidence
                total_count = self.business_metrics.confidence_count + 1
                self.business_metrics.avg_confidence_score = total_confidence / total_count
            else:
                self.business_metrics.avg_confidence_score = confidence
                self.business_metrics.confidence_sum = confidence
                self.business_metrics.confidence_count = 1
        
        # Update Prometheus metrics
        self.registry.get("plant_detector_analyses_total").inc()
        self.registry.get("plant_detector_unique_users_total").set(self.business_metrics.unique_users)
        self.registry.get("plant_detector_disease_detections_total").inc()
        self.registry.get("plant_detector_healthy_detections_total").inc()
        self.registry.get("plant_detector_avg_confidence").set(self.business_metrics.avg_confidence_score)

    async def record_alert(self, level: str, message: str, component: str = None, 
                      user_id: str = None, metadata: Dict[str, Any] = None):
        """Record system alert"""
        alert_data = {
            'level': level,
            'message': message,
            'component': component,
            'user_id': user_id,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Update alert metrics
        self.alert_metrics.total_alerts += 1
        
        if level == 'critical':
            self.alert_metrics.critical_alerts += 1
        elif level == 'warning':
            self.alert_metrics.warning_alerts += 1
        else:
            self.alert_metrics.info_alerts += 1
        
        self.alert_metrics.last_alert_time = datetime.now()
        
        # Add to history
        self.alert_history.append(alert_data)
        
        # Update Prometheus
        self.registry.get("plant_detector_alerts_total").inc()
        
        # Check for alert escalation
        await self._check_alert_conditions(alert_data)

    async def _check_alert_conditions(self, alert_data: Dict[str, Any]):
        """Check if alert conditions require escalation"""
        try:
            conditions_met = []
            
            # High error rate
            if self.system_metrics.error_rate > self.performance_thresholds.error_rate_critical_threshold:
                conditions_met.append("High error rate detected")
            
            # Low average confidence
            if self.business_metrics.avg_confidence_score < self.performance_thresholds.confidence_critical_threshold:
                conditions_met.append("Low average confidence score")
            
            # High resource usage
            if self.system_metrics.cpu_usage_percent > self.performance_thresholds.cpu_critical_threshold:
                conditions_met.append("High CPU usage")
            
            if self.system_metrics.memory_usage_percent > self.performance_thresholds.memory_critical_threshold:
                conditions_met.append("High memory usage")
            
            # Multiple critical alerts
            if self.alert_metrics.critical_alerts > 5:
                conditions_met.append("Multiple critical alerts")
            
            if conditions_met:
                escalation_message = f"System performance degradation detected: {', '.join(conditions_met)}"
                await self.record_alert('critical', escalation_message, 'system_monitor', metadata=alert_data)

        except Exception as e:
            logger.error(f"Failed to check alert conditions: {str(e)}")

    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        # System metrics monitoring
        system_task = asyncio.create_task(self._monitor_system_metrics(), name="system_monitor")
        self.monitoring_tasks.append(system_task)
        
        # Metrics persistence task
        persistence_task = asyncio.create_task(self._persist_metrics_periodically(), name="metrics_persistence")
        self.monitoring_tasks.append(persistence_task)
        
        logger.info(f"Started {len(self.monitoring_tasks)} background monitoring tasks")

    async def _monitor_system_metrics(self):
        """Background task to monitor system metrics"""
        while True:
            try:
                await self.record_system_metrics()
                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                logger.info("System monitoring task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in system monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _persist_metrics_periodically(self):
        """Background task to persist metrics periodically"""
        while True:
            try:
                await self._persist_all_metrics()
                await asyncio.sleep(300)  # Persist every 5 minutes
            except asyncio.CancelledError:
                logger.info("Metrics persistence task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in metrics persistence: {str(e)}")
                await asyncio.sleep(600)  # Wait longer on error

    async def _store_metrics_in_redis(self, model_type: str, metrics: ModelMetrics):
        """Store metrics in Redis if available"""
        if not self.redis_client:
            return
        
        try:
            key = f"{self.metrics_key_prefix}model_metrics:{model_type}"
            
            # Convert to dict for storage
            metrics_dict = {
                'model_name': metrics.model_name,
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'average_processing_time_ms': metrics.average_processing_time_ms,
                'confidence_sum': metrics.confidence_sum,
                'confidence_count': metrics.confidence_count,
                'last_request_time': metrics.last_request_time.isoformat() if metrics.last_request_time else None,
                'error_rate': metrics.error_rate,
                'uptime_percentage': metrics.uptime_percentage,
                'updated_at': datetime.now().isoformat()
            }
            
            # Store with expiration
            await self.redis_client.setex(
                key,
                json.dumps(metrics_dict),
                ex=3600  # 1 hour expiration
            )
            
        except Exception as e:
            logger.error(f"Failed to store metrics in Redis: {str(e)}")

    async def get_model_metrics(self, model_type: str = None) -> Dict[str, Any]:
        """Get model performance metrics"""
        if model_type:
            if model_type in self.model_metrics:
                metrics = self.model_metrics[model_type]
                return {
                    'model_name': metrics.model_name,
                    'total_requests': metrics.total_requests,
                    'successful_requests': metrics.successful_requests,
                    'failed_requests': metrics.failed_requests,
                    'average_processing_time_ms': metrics.average_processing_time_ms,
                    'error_rate': metrics.error_rate,
                    'uptime_percentage': metrics.uptime_percentage,
                    'last_request_time': metrics.last_request_time.isoformat() if metrics.last_request_time else None
                }
        
        # Return all model metrics if no specific type requested
        return {
            model_type: {
                'model_name': metrics.model_name,
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'average_processing_time_ms': metrics.average_processing_time_ms,
                'error_rate': metrics.error_rate,
                'uptime_percentage': metrics.uptime_percentage,
                    'last_request_time': metrics.last_request_time.isoformat() if metrics.last_request_time else None
            }
            for model_type, metrics in self.model_metrics.items()
        }

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        return {
            'cpu_usage_percent': self.system_metrics.cpu_usage_percent,
            'memory_usage_percent': self.system_metrics.memory_usage_percent,
            'disk_usage_percent': self.system_metrics.disk_usage_percent,
            'network_io': self.system_metrics.network_io,
            'active_connections': self.system_metrics.active_connections,
            'uptime_seconds': self.system_metrics.uptime_seconds,
            'error_count': self.system_metrics.error_count,
            'request_rate_per_minute': self.system_metrics.request_rate_per_minute,
            'system_health': 'healthy' if self.system_metrics.error_count == 0 else 'degraded',
            'updated_at': datetime.now().isoformat()
        }

    async def get_business_metrics(self) -> Dict[str, Any]:
        """Get business intelligence metrics"""
        return {
            'total_analyses': self.business_metrics.total_analyses,
            'unique_users': self.business_metrics.unique_users,
            'disease_detections': self.business_metrics.disease_detections,
            'healthy_detections': self.business_metrics.healthy_detections,
            'top_diseases': dict(sorted(self.business_metrics.top_diseases.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]),
            'top_plants': dict(sorted(self.business_metrics.top_plants.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]),
            'avg_confidence_score': self.business_metrics.avg_confidence_score,
            'user_satisfaction_score': self.business_metrics.user_satisfaction_score,
            'analysis_success_rate': (self.business_metrics.total_analyses - self.system_metrics.error_count) / max(1, self.business_metrics.total_analyses),
            'updated_at': datetime.now().isoformat()
        }

    async def get_alert_metrics(self, limit: int = 100) -> Dict[str, Any]:
        """Get alert metrics and recent alerts"""
        recent_alerts = list(self.alert_history)[-limit:] if self.alert_history else []
        
        return {
            'total_alerts': self.alert_metrics.total_alerts,
            'critical_alerts': self.alert_metrics.critical_alerts,
            'warning_alerts': self.alert_metrics.warning_alerts,
            'info_alerts': self.alert_metrics.info_alerts,
            'last_alert_time': self.alert_metrics.last_alert_time.isoformat() if self.alert_metrics.last_alert_time else None,
            'recent_alerts': recent_alerts,
            'alert_rate_per_hour': len([a for a in recent_alerts 
                                   if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)]),
            'active_alerts': self.alert_metrics.active_alerts,
            'updated_at': datetime.now().isoformat()
        }

    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        return {
            'model_performance': await self.get_model_metrics(),
            'system_performance': await self.get_system_metrics(),
            'business_metrics': await self.get_business_metrics(),
            'alert_metrics': await self.get_alert_metrics(),
            'performance_health': {
                'overall_health': 'healthy' if self.system_metrics.error_count == 0 else 'degraded',
                'model_health': 'good' if all(m.error_rate < 0.05 for m in self.model_metrics.values()) else 'needs_attention',
                'system_resources': 'good' if (self.system_metrics.cpu_usage_percent < 70 and 
                                            self.system_metrics.memory_usage_percent < 75) else 'needs_attention',
                'response_times': 'good' if all(m.average_processing_time_ms < 1000 for m in self.model_metrics.values()) else 'needs_improvement'
            },
            'updated_at': datetime.now().isoformat()
        }

    async def push_metrics_to_gateway(self):
        """Push current metrics to Prometheus gateway"""
        try:
            # This would be configured for your Prometheus pushgateway
            # Implementation depends on your specific gateway setup
            push_to_gateway(
                self.registry,
                job='plant_detector_metrics',
                grouping='plant_disease_detector',
                registry='default'
            )
            logger.info("Metrics pushed to Prometheus gateway")
        except Exception as e:
            logger.error(f"Failed to push metrics to gateway: {str(e)}")

    def add_alert_subscriber(self, callback):
        """Add alert subscriber for notifications"""
        self.alert_subscribers.append(callback)

    def remove_alert_subscriber(self, callback):
        """Remove alert subscriber"""
        if callback in self.alert_subscribers:
            self.alert_subscribers.remove(callback)

    async def notify_subscribers(self, alert_data: Dict[str, Any]):
        """Notify all alert subscribers"""
        for subscriber in self.alert_subscribers:
            try:
                await subscriber(alert_data)
            except Exception as e:
                logger.error(f"Failed to notify subscriber: {str(e)}")

    async def generate_performance_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_range_hours)
        
        # Calculate performance improvements (if historical data available)
        report_data = {
            'report_period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'hours': time_range_hours
            },
            'summary': {
                'total_requests': sum(m.total_requests for m in self.model_metrics.values()),
                'successful_requests': sum(m.successful_requests for m in self.model_metrics.values()),
                'failed_requests': sum(m.failed_requests for m in self.model_metrics.values()),
                'average_success_rate': 0.0,
                'system_uptime_percentage': ((end_time.timestamp() - self.start_time) / 3600) * 100
            },
            'model_breakdown': {
                model_type.value: {
                    'total_requests': metrics.total_requests,
                    'success_rate': metrics.successful_requests / max(1, metrics.total_requests) if metrics.total_requests > 0 else 0,
                    'average_processing_time_ms': metrics.average_processing_time_ms,
                    'error_rate': metrics.error_rate,
                    'uptime_percentage': metrics.uptime_percentage
                }
                for model_type, metrics in self.model_metrics.items()
            },
            'system_health': {
                'peak_cpu_usage': getattr(self, '_peak_cpu', 0),
                'peak_memory_usage': getattr(self, '_peak_memory', 0),
                'total_errors': self.system_metrics.error_count,
                'resource_utilization': 'normal' if self.system_metrics.cpu_usage_percent < 70 else 'high'
            },
            'business_insights': {
                'total_analyses': self.business_metrics.total_analyses,
                'unique_users': self.business_metrics.unique_users,
                'disease_detection_rate': self.business_metrics.disease_detections / max(1, self.business_metrics.total_analyses) if self.business_metrics.total_analyses > 0 else 0,
                'top_detected_diseases': dict(sorted(self.business_metrics.top_diseases.items(), 
                                                  key=lambda x: x[1], reverse=True)[:5]),
                'average_confidence': self.business_metrics.avg_confidence_score
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report_data

    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on current metrics"""
        recommendations = []
        
        # Model performance recommendations
        for model_type, metrics in self.model_metrics.items():
            if metrics.error_rate > self.performance_thresholds.error_rate_warning_threshold:
                recommendations.append(f"Improve {model_type.value} model reliability - current error rate: {metrics.error_rate:.2%}")
            
            if metrics.average_processing_time_ms > self.performance_thresholds.response_time_warning_threshold:
                recommendations.append(f"Optimize {model_type.value} model performance - current avg time: {metrics.average_processing_time_ms:.0f}ms")
        
        # System resource recommendations
        if self.system_metrics.cpu_usage_percent > self.performance_thresholds.cpu_warning_threshold:
            recommendations.append("Consider scaling CPU resources or optimizing algorithms")
        
        if self.system_metrics.memory_usage_percent > self.performance_thresholds.memory_warning_threshold:
            recommendations.append("Monitor memory usage and consider optimization strategies")
        
        # Business recommendations
        if self.business_metrics.avg_confidence_score < self.performance_thresholds.confidence_warning_threshold:
            recommendations.append("Review AI model confidence scores and calibration")
        
        if self.alert_metrics.critical_alerts > 3:
            recommendations.append("Review system stability and error handling procedures")
        
        return recommendations