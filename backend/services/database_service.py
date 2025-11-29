"""
Database Service

This module provides comprehensive database integration for the Plant Leaf Disease Detection System.
It supports PostgreSQL for production data and Redis for caching and session management.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

import asyncpg
import aioredis
from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime, Float, Integer, Text, Boolean, Index
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pydantic import BaseModel
import redis.exceptions

from backend.services.config import Settings

logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()


# Database Models
class AnalysisRecord(Base):
    """Model for storing analysis results"""
    __tablename__ = "analysis_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    image_hash = Column(String, index=True)
    image_size = Column(Integer)
    image_format = Column(String)
    plant_name = Column(String, index=True)
    plant_confidence = Column(Float)
    disease_name = Column(String, index=True)
    disease_confidence = Column(Float)
    ai_models_used = Column(JSONB)
    processing_time_ms = Column(Integer)
    image_quality_score = Column(Float)
    user_ip = Column(String)
    user_agent = Column(Text)
    success = Column(Boolean, default=True, index=True)
    error_message = Column(Text)
    
    # Create indexes
    __table_args__ = (
        Index('idx_analysis_timestamp_plant', 'timestamp', 'plant_name'),
        Index('idx_analysis_disease_success', 'disease_name', 'success'),
    )


class UserSession(Base):
    """Model for storing user sessions"""
    __tablename__ = "user_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    analysis_count = Column(Integer, default=0)
    user_ip = Column(String)
    user_agent = Column(Text)
    is_active = Column(Boolean, default=True, index=True)
    session_data = Column(JSONB)


class FeedbackRecord(Base):
    """Model for storing user feedback"""
    __tablename__ = "feedback_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    feedback_type = Column(String)  # 'positive', 'negative', 'correction'
    original_plant = Column(String)
    original_disease = Column(String)
    corrected_plant = Column(String)
    corrected_disease = Column(String)
    feedback_text = Column(Text)
    rating = Column(Integer)  # 1-5 stars
    helpful = Column(Boolean)
    additional_comments = Column(Text)


class AnalyticsEvent(Base):
    """Model for storing analytics events"""
    __tablename__ = "analytics_events"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    event_type = Column(String, nullable=False, index=True)  # 'analysis', 'page_view', 'download', etc.
    session_id = Column(String, index=True)
    user_ip = Column(String)
    user_agent = Column(Text)
    event_data = Column(JSONB)
    processing_time_ms = Column(Integer)


class DatabaseService:
    """Comprehensive database service with PostgreSQL and Redis support"""
    
    def __init__(self, settings: Settings):
        """Initialize database service with settings"""
        self.settings = settings
        self.postgres_engine = None
        self.postgres_session_factory = None
        self.redis_client = None
        self.redis_pool = None
        self.is_initialized = False
        
        # Connection configuration
        self.postgres_url = getattr(settings, 'POSTGRES_URL', 'postgresql://localhost/plantdetection')
        self.redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
        
        # Pool settings
        self.postgres_pool_size = getattr(settings, 'POSTGRES_POOL_SIZE', 10)
        self.postgres_max_overflow = getattr(settings, 'POSTGRES_MAX_OVERFLOW', 20)
        self.redis_pool_size = getattr(settings, 'REDIS_POOL_SIZE', 10)
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            logger.info("Initializing database connections...")
            
            # Initialize PostgreSQL
            await self._initialize_postgres()
            
            # Initialize Redis
            await self._initialize_redis()
            
            self.is_initialized = True
            logger.info("Database service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database service: {str(e)}")
            raise
    
    async def _initialize_postgres(self):
        """Initialize PostgreSQL connection"""
        try:
            # Create async engine
            self.postgres_engine = create_async_engine(
                self.postgres_url,
                pool_size=self.postgres_pool_size,
                max_overflow=self.postgres_max_overflow,
                pool_pre_ping=True,
                echo=False
            )
            
            # Create session factory
            self.postgres_session_factory = async_sessionmaker(
                self.postgres_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            await self._create_tables()
            
            logger.info("PostgreSQL connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {str(e)}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            # Create Redis connection pool
            self.redis_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.redis_pool_size,
                retry_on_timeout=True,
                decode_responses=True
            )
            
            # Create Redis client
            self.redis_client = aioredis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create database tables"""
        try:
            async with self.postgres_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created/verified")
            
        except Exception as e:
            logger.error(f"Failed to create database tables: {str(e)}")
            raise
    
    async def store_analysis_result(self, analysis_data: Dict[str, Any]) -> str:
        """Store analysis result in database"""
        try:
            analysis_id = analysis_data.get('session_id') or str(uuid.uuid4())
            
            # Store in PostgreSQL
            async with self.postgres_session_factory() as session:
                analysis_record = AnalysisRecord(
                    id=analysis_id,
                    session_id=analysis_data.get('session_id'),
                    timestamp=datetime.utcnow(),
                    image_hash=analysis_data.get('metadata', {}).get('image_hash'),
                    image_size=analysis_data.get('metadata', {}).get('file_size'),
                    image_format=analysis_data.get('metadata', {}).get('image_format'),
                    plant_name=analysis_data.get('plant_detection', {}).get('plant_name'),
                    plant_confidence=analysis_data.get('plant_detection', {}).get('confidence'),
                    disease_name=analysis_data.get('disease_detection', {}).get('primary_disease', {}).get('disease_name'),
                    disease_confidence=analysis_data.get('disease_detection', {}).get('primary_disease', {}).get('confidence'),
                    ai_models_used=analysis_data.get('ai_model_results'),
                    processing_time_ms=analysis_data.get('metadata', {}).get('processing_time_total_ms'),
                    image_quality_score=analysis_data.get('metadata', {}).get('image_quality_score'),
                    user_ip=analysis_data.get('user_ip'),
                    user_agent=analysis_data.get('user_agent'),
                    success=analysis_data.get('success', True),
                    error_message=analysis_data.get('error_message')
                )
                
                session.add(analysis_record)
                await session.commit()
            
            # Store recent result in Redis cache
            await self._cache_analysis_result(analysis_id, analysis_data, ttl=3600)  # 1 hour
            
            logger.info(f"Analysis result stored: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Failed to store analysis result: {str(e)}")
            raise
    
    async def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis result from database"""
        try:
            # Try Redis cache first
            cached_result = await self._get_cached_analysis_result(analysis_id)
            if cached_result:
                return cached_result
            
            # Fallback to PostgreSQL
            async with self.postgres_session_factory() as session:
                from sqlalchemy import select
                
                result = await session.execute(
                    select(AnalysisRecord).where(AnalysisRecord.id == analysis_id)
                )
                record = result.scalar_one_or_none()
                
                if record:
                    analysis_data = {
                        'id': record.id,
                        'session_id': record.session_id,
                        'timestamp': record.timestamp.isoformat(),
                        'plant_detection': {
                            'plant_name': record.plant_name,
                            'confidence': record.plant_confidence
                        },
                        'disease_detection': {
                            'primary_disease': {
                                'disease_name': record.disease_name,
                                'confidence': record.disease_confidence
                            } if record.disease_name else None
                        },
                        'ai_model_results': record.ai_models_used,
                        'metadata': {
                            'image_quality_score': record.image_quality_score,
                            'processing_time_total_ms': record.processing_time_ms,
                            'image_format': record.image_format
                        },
                        'success': record.success,
                        'error_message': record.error_message
                    }
                    
                    # Cache the result
                    await self._cache_analysis_result(analysis_id, analysis_data, ttl=1800)  # 30 min
                    
                    return analysis_data
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get analysis result {analysis_id}: {str(e)}")
            return None
    
    async def store_user_session(self, session_data: Dict[str, Any]) -> str:
        """Store user session data"""
        try:
            session_id = session_data.get('session_id') or str(uuid.uuid4())
            
            async with self.postgres_session_factory() as session:
                user_session = UserSession(
                    session_id=session_id,
                    created_at=datetime.utcnow(),
                    last_activity=datetime.utcnow(),
                    analysis_count=session_data.get('analysis_count', 0),
                    user_ip=session_data.get('user_ip'),
                    user_agent=session_data.get('user_agent'),
                    is_active=True,
                    session_data=session_data.get('additional_data', {})
                )
                
                session.merge(user_session)  # Use merge to handle existing sessions
                await session.commit()
            
            # Also store in Redis for fast access
            await self._cache_session_data(session_id, session_data, ttl=86400)  # 24 hours
            
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to store user session: {str(e)}")
            raise
    
    async def get_user_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get user session data"""
        try:
            # Try Redis cache first
            cached_session = await self._get_cached_session_data(session_id)
            if cached_session:
                return cached_session
            
            # Fallback to PostgreSQL
            async with self.postgres_session_factory() as session:
                from sqlalchemy import select
                
                result = await session.execute(
                    select(UserSession).where(UserSession.session_id == session_id, UserSession.is_active == True)
                )
                record = result.scalar_one_or_none()
                
                if record:
                    session_data = {
                        'session_id': record.session_id,
                        'created_at': record.created_at.isoformat(),
                        'last_activity': record.last_activity.isoformat(),
                        'analysis_count': record.analysis_count,
                        'user_ip': record.user_ip,
                        'user_agent': record.user_agent,
                        'additional_data': record.session_data
                    }
                    
                    # Cache the session
                    await self._cache_session_data(session_id, session_data, ttl=3600)  # 1 hour
                    
                    return session_data
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user session {session_id}: {str(e)}")
            return None
    
    async def store_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Store user feedback"""
        try:
            feedback_id = str(uuid.uuid4())
            
            async with self.postgres_session_factory() as session:
                feedback_record = FeedbackRecord(
                    id=feedback_id,
                    analysis_id=feedback_data.get('analysis_id'),
                    session_id=feedback_data.get('session_id'),
                    timestamp=datetime.utcnow(),
                    feedback_type=feedback_data.get('feedback_type'),
                    original_plant=feedback_data.get('original_plant'),
                    original_disease=feedback_data.get('original_disease'),
                    corrected_plant=feedback_data.get('corrected_plant'),
                    corrected_disease=feedback_data.get('corrected_disease'),
                    feedback_text=feedback_data.get('feedback_text'),
                    rating=feedback_data.get('rating'),
                    helpful=feedback_data.get('helpful'),
                    additional_comments=feedback_data.get('additional_comments')
                )
                
                session.add(feedback_record)
                await session.commit()
            
            logger.info(f"Feedback stored: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {str(e)}")
            raise
    
    async def store_analytics_event(self, event_data: Dict[str, Any]) -> str:
        """Store analytics event"""
        try:
            event_id = str(uuid.uuid4())
            
            async with self.postgres_session_factory() as session:
                analytics_record = AnalyticsEvent(
                    id=event_id,
                    timestamp=datetime.utcnow(),
                    event_type=event_data.get('event_type'),
                    session_id=event_data.get('session_id'),
                    user_ip=event_data.get('user_ip'),
                    user_agent=event_data.get('user_agent'),
                    event_data=event_data.get('event_data', {}),
                    processing_time_ms=event_data.get('processing_time_ms')
                )
                
                session.add(analytics_record)
                await session.commit()
            
            # Also store recent events in Redis for real-time analytics
            await self._store_analytics_in_redis(event_data)
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to store analytics event: {str(e)}")
            raise
    
    async def get_analytics_data(self, event_type: Optional[str] = None, 
                               time_range_hours: int = 24) -> List[Dict[str, Any]]:
        """Get analytics data with optional filtering"""
        try:
            async with self.postgres_session_factory() as session:
                from sqlalchemy import select, and_
                
                # Build query
                query = select(AnalyticsEvent).where(
                    AnalyticsEvent.timestamp >= datetime.utcnow() - timedelta(hours=time_range_hours)
                )
                
                if event_type:
                    query = query.where(AnalyticsEvent.event_type == event_type)
                
                query = query.order_by(AnalyticsEvent.timestamp.desc())
                
                result = await session.execute(query)
                records = result.scalars().all()
                
                analytics_data = []
                for record in records:
                    analytics_data.append({
                        'id': record.id,
                        'timestamp': record.timestamp.isoformat(),
                        'event_type': record.event_type,
                        'session_id': record.session_id,
                        'user_ip': record.user_ip,
                        'event_data': record.event_data,
                        'processing_time_ms': record.processing_time_ms
                    })
                
                return analytics_data
                
        except Exception as e:
            logger.error(f"Failed to get analytics data: {str(e)}")
            return []
    
    async def get_analytics_summary(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get analytics summary statistics"""
        try:
            async with self.postgres_session_factory() as session:
                from sqlalchemy import select, func, and_
                
                # Get base date
                base_date = datetime.utcnow() - timedelta(hours=time_range_hours)
                
                # Total events
                total_events_result = await session.execute(
                    select(func.count(AnalyticsEvent.id)).where(
                        AnalyticsEvent.timestamp >= base_date
                    )
                )
                total_events = total_events_result.scalar() or 0
                
                # Events by type
                events_by_type_result = await session.execute(
                    select(
                        AnalyticsEvent.event_type,
                        func.count(AnalyticsEvent.id).label('count')
                    ).where(
                        AnalyticsEvent.timestamp >= base_date
                    ).group_by(AnalyticsEvent.event_type)
                )
                events_by_type = {
                    row.event_type: row.count for row in events_by_type_result
                }
                
                # Analysis success rate
                analysis_success_result = await session.execute(
                    select(
                        func.count(AnalysisRecord.id).label('total'),
                        func.sum(func.cast(AnalysisRecord.success, Integer)).label('successful')
                    ).where(
                        AnalysisRecord.timestamp >= base_date
                    )
                )
                analysis_stats = analysis_success_result.first()
                success_rate = (
                    (analysis_stats.successful / analysis_stats.total * 100) 
                    if analysis_stats.total > 0 else 0
                )
                
                # Top plants detected
                top_plants_result = await session.execute(
                    select(
                        AnalysisRecord.plant_name,
                        func.count(AnalysisRecord.id).label('count')
                    ).where(
                        and_(
                            AnalysisRecord.timestamp >= base_date,
                            AnalysisRecord.plant_name.isnot(None)
                        )
                    ).group_by(AnalysisRecord.plant_name).order_by(
                        func.count(AnalysisRecord.id).desc()
                    ).limit(10)
                )
                top_plants = {
                    row.plant_name: row.count for row in top_plants_result
                }
                
                # Top diseases detected
                top_diseases_result = await session.execute(
                    select(
                        AnalysisRecord.disease_name,
                        func.count(AnalysisRecord.id).label('count')
                    ).where(
                        and_(
                            AnalysisRecord.timestamp >= base_date,
                            AnalysisRecord.disease_name.isnot(None)
                        )
                    ).group_by(AnalysisRecord.disease_name).order_by(
                        func.count(AnalysisRecord.id).desc()
                    ).limit(10)
                )
                top_diseases = {
                    row.disease_name: row.count for row in top_diseases_result
                }
                
                return {
                    'time_range_hours': time_range_hours,
                    'total_events': total_events,
                    'events_by_type': events_by_type,
                    'analysis_stats': {
                        'total_analyses': analysis_stats.total or 0,
                        'successful_analyses': analysis_stats.successful or 0,
                        'success_rate': round(success_rate, 2)
                    },
                    'top_plants_detected': top_plants,
                    'top_diseases_detected': top_diseases
                }
                
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {str(e)}")
            return {}
    
    async def _cache_analysis_result(self, analysis_id: str, data: Dict[str, Any], ttl: int = 3600):
        """Cache analysis result in Redis"""
        try:
            if self.redis_client:
                cache_key = f"analysis:{analysis_id}"
                await self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(data, default=str)
                )
        except Exception as e:
            logger.warning(f"Failed to cache analysis result: {str(e)}")
    
    async def _get_cached_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result from Redis"""
        try:
            if self.redis_client:
                cache_key = f"analysis:{analysis_id}"
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached analysis result: {str(e)}")
            return None
    
    async def _cache_session_data(self, session_id: str, data: Dict[str, Any], ttl: int = 3600):
        """Cache session data in Redis"""
        try:
            if self.redis_client:
                cache_key = f"session:{session_id}"
                await self.redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(data, default=str)
                )
        except Exception as e:
            logger.warning(f"Failed to cache session data: {str(e)}")
    
    async def _get_cached_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session data from Redis"""
        try:
            if self.redis_client:
                cache_key = f"session:{session_id}"
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            return None
        except Exception as e:
            logger.warning(f"Failed to get cached session data: {str(e)}")
            return None
    
    async def _store_analytics_in_redis(self, event_data: Dict[str, Any]):
        """Store analytics event in Redis for real-time processing"""
        try:
            if self.redis_client:
                # Store in a Redis list for recent events
                event_key = f"analytics:recent:{event_data.get('event_type', 'general')}"
                await self.redis_client.lpush(
                    event_key,
                    json.dumps({**event_data, 'timestamp': datetime.utcnow().isoformat()}, default=str)
                )
                
                # Keep only recent 1000 events
                await self.redis_client.ltrim(event_key, 0, 999)
                
                # Set expiration (24 hours)
                await self.redis_client.expire(event_key, 86400)
                
        except Exception as e:
            logger.warning(f"Failed to store analytics in Redis: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            health_status = True
            
            # Check PostgreSQL
            try:
                async with self.postgres_session_factory() as session:
                    await session.execute("SELECT 1")
                logger.debug("PostgreSQL health check passed")
            except Exception as e:
                logger.error(f"PostgreSQL health check failed: {str(e)}")
                health_status = False
            
            # Check Redis
            try:
                if self.redis_client:
                    await self.redis_client.ping()
                    logger.debug("Redis health check passed")
                else:
                    logger.warning("Redis client not initialized")
            except Exception as e:
                logger.error(f"Redis health check failed: {str(e)}")
                health_status = False
            
            return health_status
            
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {}
            
            # PostgreSQL stats
            async with self.postgres_session_factory() as session:
                from sqlalchemy import select, func
                
                # Table sizes
                stats['postgresql'] = {
                    'analysis_records': await session.scalar(select(func.count(AnalysisRecord.id))),
                    'user_sessions': await session.scalar(select(func.count(UserSession.id))),
                    'feedback_records': await session.scalar(select(func.count(FeedbackRecord.id))),
                    'analytics_events': await session.scalar(select(func.count(AnalyticsEvent.id)))
                }
            
            # Redis stats
            if self.redis_client:
                try:
                    redis_info = await self.redis_client.info()
                    stats['redis'] = {
                        'connected_clients': redis_info.get('connected_clients', 0),
                        'used_memory': redis_info.get('used_memory_human', '0B'),
                        'total_commands_processed': redis_info.get('total_commands_processed', 0),
                        'keyspace_hits': redis_info.get('keyspace_hits', 0),
                        'keyspace_misses': redis_info.get('keyspace_misses', 0)
                    }
                except Exception as e:
                    logger.warning(f"Failed to get Redis stats: {str(e)}")
                    stats['redis'] = {'error': str(e)}
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {str(e)}")
            return {}
    
    async def cleanup(self):
        """Cleanup database connections"""
        try:
            logger.info("Cleaning up database connections...")
            
            # Cleanup PostgreSQL
            if self.postgres_engine:
                await self.postgres_engine.dispose()
                logger.info("PostgreSQL connection closed")
            
            # Cleanup Redis
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis connection closed")
            
            if self.redis_pool:
                await self.redis_pool.disconnect()
                logger.info("Redis pool closed")
            
            logger.info("Database connections cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup database connections: {str(e)}")


# Import uuid for UUID generation
import uuid


# Factory function for easy database service creation
def create_database_service(settings: Settings) -> DatabaseService:
    """Factory function to create a database service"""
    return DatabaseService(settings)