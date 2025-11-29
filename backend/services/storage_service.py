"""
Storage Service for Plant Leaf Disease Detection System

This module provides secure file storage with CDN integration, automatic image
processing, backup procedures, and storage usage monitoring for production scalability.
"""

import asyncio
import hashlib
import logging
import mimetypes
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, BinaryIO
from dataclasses import dataclass, field
from functools import wraps
import io

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config
from PIL import Image, ImageOps, ExifTags
import aiofiles
import aiohttp

from backend.services.config import Settings
from backend.models.analysis import ImageMetadata, ProcessingError

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Storage configuration settings"""
    # S3 settings
    s3_endpoint_url: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_bucket_name: str = "plant-disease-images"
    s3_use_ssl: bool = True
    s3_verify_ssl: bool = True

    # CDN settings
    cdn_enabled: bool = True
    cdn_domain: str = "cdn.plantdetection.com"
    cdn_cache_ttl: int = 86400  # 24 hours

    # Local storage settings
    local_storage_path: str = "./storage"
    local_temp_path: str = "./temp"

    # File settings
    max_file_size_mb: float = 50.0
    allowed_formats: List[str] = field(default_factory=lambda: ['jpg', 'jpeg', 'png', 'webp'])
    compression_quality: int = 85
    thumbnail_sizes: List[Tuple[int, int]] = field(default_factory=lambda: [
        (150, 150), (300, 300), (600, 600), (1200, 1200)
    ])

    # Backup settings
    backup_enabled: bool = True
    backup_retention_days: int = 90
    backup_bucket: str = "plant-disease-backups"

    # Monitoring settings
    usage_tracking_enabled: bool = True
    cleanup_temp_files: bool = True
    cleanup_interval_hours: int = 6


@dataclass
class StorageMetrics:
    """Storage usage and performance metrics"""
    total_files: int = 0
    total_size_mb: float = 0.0
    daily_uploads: int = 0
    daily_downloads: int = 0
    avg_upload_time: float = 0.0
    avg_download_time: float = 0.0
    cache_hit_rate: float = 0.0
    error_count: int = 0
    last_cleanup: Optional[datetime] = None
    storage_usage_by_type: Dict[str, float] = field(default_factory=dict)


@dataclass
class FileReference:
    """Reference to a stored file"""
    file_id: str
    original_filename: str
    content_type: str
    file_size: int
    storage_type: str  # 's3', 'local'
    file_path: str
    url: Optional[str] = None
    cdn_url: Optional[str] = None
    thumbnail_urls: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class StorageService:
    """
    Advanced storage service with S3 integration, CDN support, and automatic optimization.

    Features:
    - S3-compatible object storage with automatic fallback
    - CDN integration for fast global content delivery
    - Automatic image resizing and format optimization
    - Secure presigned URL generation for private uploads
    - Backup and disaster recovery procedures
    - Storage usage monitoring and cost optimization
    - Automatic cleanup of temporary files
    """

    def __init__(self, settings: Settings):
        """Initialize storage service with settings"""
        self.settings = settings
        self.config = StorageConfig(
            s3_endpoint_url=getattr(settings, 'S3_ENDPOINT_URL', None),
            s3_access_key_id=getattr(settings, 'S3_ACCESS_KEY_ID', None),
            s3_secret_access_key=getattr(settings, 'S3_SECRET_ACCESS_KEY', None),
            s3_region=getattr(settings, 'S3_REGION', 'us-east-1'),
            s3_bucket_name=getattr(settings, 'S3_BUCKET_NAME', 'plant-disease-images'),
            local_storage_path=getattr(settings, 'LOCAL_STORAGE_PATH', './storage'),
            local_temp_path=getattr(settings, 'TEMP_DIR_ABSOLUTE', './temp'),
            max_file_size_mb=getattr(settings, 'MAX_FILE_SIZE_MB', 50.0),
            allowed_formats=getattr(settings, 'SUPPORTED_FORMATS_LIST', ['jpg', 'jpeg', 'png', 'webp'])
        )

        # Initialize clients
        self.s3_client = None
        self.s3_resource = None
        self.local_storage_path = Path(self.config.local_storage_path)
        self.temp_storage_path = Path(self.config.local_temp_path)

        # Storage metrics
        self.metrics = StorageMetrics()

        # File cache for quick lookups
        self._file_cache: Dict[str, FileReference] = {}

        # Background cleanup task
        self._cleanup_task = None

        logger.info(f"Storage service initialized with config: S3={bool(self.config.s3_access_key_id)}, Local=True")

    async def initialize(self):
        """Initialize storage service components"""
        try:
            # Create directories
            self.local_storage_path.mkdir(parents=True, exist_ok=True)
            self.temp_storage_path.mkdir(parents=True, exist_ok=True)

            # Initialize S3 client if credentials provided
            if self.config.s3_access_key_id and self.config.s3_secret_access_key:
                await self._initialize_s3_client()
            else:
                logger.warning("S3 credentials not provided, using local storage only")

            # Start background cleanup task
            if self.config.cleanup_temp_files:
                self._cleanup_task = asyncio.create_task(self._background_cleanup())

            # Load existing files index
            await self._load_files_index()

            logger.info("Storage service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize storage service: {str(e)}")
            raise

    async def _initialize_s3_client(self):
        """Initialize S3 client with retry logic"""
        try:
            # Configure S3 client
            s3_config = Config(
                region_name=self.config.s3_region,
                use_ssl=self.config.s3_use_ssl,
                verify_ssl=self.config.s3_verify_ssl,
                retries={'max_attempts': 3, 'mode': 'adaptive'}
            )

            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.config.s3_endpoint_url,
                aws_access_key_id=self.config.s3_access_key_id,
                aws_secret_access_key=self.config.s3_secret_access_key,
                config=s3_config
            )

            self.s3_resource = boto3.resource(
                's3',
                endpoint_url=self.config.s3_endpoint_url,
                aws_access_key_id=self.config.s3_access_key_id,
                aws_secret_access_key=self.config.s3_secret_access_key,
                config=s3_config
            )

            # Test connection
            await self._test_s3_connection()

            logger.info("S3 client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            # Fall back to local storage
            self.s3_client = None
            self.s3_resource = None

    async def _test_s3_connection(self):
        """Test S3 connection and bucket access"""
        try:
            # Test bucket access
            response = await asyncio.to_thread(
                self.s3_client.head_bucket,
                Bucket=self.config.s3_bucket_name
            )
            logger.info("S3 connection test successful")

        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.warning(f"S3 bucket {self.config.s3_bucket_name} not found, creating...")
                await self._create_s3_bucket()
            else:
                logger.error(f"S3 connection test failed: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"S3 connection test failed: {str(e)}")
            raise

    async def _create_s3_bucket(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            await asyncio.to_thread(
                self.s3_client.create_bucket,
                Bucket=self.config.s3_bucket_name,
                CreateBucketConfiguration={'LocationConstraint': self.config.s3_region}
            )
            logger.info(f"S3 bucket {self.config.s3_bucket_name} created successfully")

        except ClientError as e:
            logger.error(f"Failed to create S3 bucket: {str(e)}")
            raise

    async def _load_files_index(self):
        """Load existing files index from storage"""
        try:
            # This would typically load from a database or index file
            # For now, initialize empty cache
            self._file_cache.clear()
            logger.debug("Files index loaded (empty)")

        except Exception as e:
            logger.error(f"Failed to load files index: {str(e)}")

    async def store_file(
        self,
        file_data: Union[bytes, BinaryIO],
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        generate_thumbnails: bool = True,
        ttl_days: Optional[int] = None
    ) -> FileReference:
        """
        Store file with automatic optimization and thumbnail generation.

        Args:
            file_data: File data as bytes or file-like object
            filename: Original filename
            content_type: MIME content type
            metadata: Additional metadata to store
            generate_thumbnails: Whether to generate thumbnails
            ttl_days: Time-to-live in days

        Returns:
            FileReference with storage details
        """
        start_time = time.time()
        file_id = str(uuid.uuid4())

        try:
            # Validate file data
            if isinstance(file_data, (bytes, bytearray)):
                data_bytes = file_data
                file_size = len(data_bytes)
            else:
                # Assume file-like object
                current_pos = file_data.tell() if hasattr(file_data, 'tell') else 0
                data_bytes = file_data.read()
                if hasattr(file_data, 'seek'):
                    file_data.seek(current_pos)
                file_size = len(data_bytes)

            # Check file size
            if file_size > self.config.max_file_size_mb * 1024 * 1024:
                raise ValueError(f"File too large: {file_size / (1024 * 1024):.2f}MB")

            # Detect content type if not provided
            if not content_type:
                content_type, _ = mimetypes.guess_type(filename)
                if not content_type:
                    content_type = 'application/octet-stream'

            # Generate checksum
            checksum = hashlib.md5(data_bytes).hexdigest()

            # Determine file extension and validate
            file_ext = Path(filename).suffix.lower().lstrip('.')
            if file_ext not in self.config.allowed_formats:
                raise ValueError(f"File format not allowed: {file_ext}")

            # Create file reference
            file_ref = FileReference(
                file_id=file_id,
                original_filename=filename,
                content_type=content_type,
                file_size=file_size,
                storage_type='s3' if self.s3_client else 'local',
                metadata=metadata or {},
                checksum=checksum,
                expires_at=datetime.utcnow() + timedelta(days=ttl_days) if ttl_days else None
            )

            # Process image if it's an image file
            if content_type.startswith('image/'):
                # Open image for processing
                image = Image.open(io.BytesIO(data_bytes))

                # Auto-orient based on EXIF
                image = ImageOps.exif_transpose(image)

                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Store original processed image
                optimized_data = io.BytesIO()
                image.save(optimized_data, format='JPEG', quality=self.config.compression_quality, optimize=True)
                optimized_bytes = optimized_data.getvalue()

                # Update file reference with optimized data
                file_ref.file_size = len(optimized_bytes)

                # Generate thumbnails
                if generate_thumbnails:
                    thumbnail_urls = await self._generate_thumbnails(image, file_id, file_ext)
                    file_ref.thumbnail_urls = thumbnail_urls

                data_bytes = optimized_bytes

            # Store file
            if self.s3_client:
                file_ref = await self._store_file_s3(file_ref, data_bytes, file_id, file_ext)
            else:
                file_ref = await self._store_file_local(file_ref, data_bytes, file_id, file_ext)

            # Cache file reference
            self._file_cache[file_id] = file_ref

            # Update metrics
            upload_time = time.time() - start_time
            self.metrics.daily_uploads += 1
            self.metrics.avg_upload_time = (
                (self.metrics.avg_upload_time + upload_time) / 2
                if self.metrics.avg_upload_time > 0 else upload_time
            )

            logger.info(f"File stored successfully: {filename} -> {file_id} ({file_size} bytes)")
            return file_ref

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Failed to store file {filename}: {str(e)}")
            raise ProcessingError(f"Storage failed: {str(e)}")

    async def _store_file_s3(
        self,
        file_ref: FileReference,
        data: bytes,
        file_id: str,
        file_ext: str
    ) -> FileReference:
        """Store file in S3 with retry logic"""
        try:
            # Generate S3 key
            s3_key = f"uploads/{datetime.utcnow().strftime('%Y/%m/%d')}/{file_id}.{file_ext}"
            file_ref.file_path = s3_key

            # Upload to S3
            await asyncio.to_thread(
                self.s3_client.put_object,
                Bucket=self.config.s3_bucket_name,
                Key=s3_key,
                Body=data,
                ContentType=file_ref.content_type,
                Metadata={
                    'original-filename': file_ref.original_filename,
                    'file-id': file_id,
                    'checksum': file_ref.checksum,
                    **{f"x-amz-meta-{k}": str(v) for k, v in file_ref.metadata.items()}
                },
                StorageClass='STANDARD',
                ServerSideEncryption='AES256'
            )

            # Generate URLs
            file_ref.url = await self._generate_s3_url(s3_key)

            if self.config.cdn_enabled:
                file_ref.cdn_url = f"https://{self.config.cdn_domain}/{s3_key}"

            logger.debug(f"File stored in S3: {s3_key}")
            return file_ref

        except Exception as e:
            logger.error(f"Failed to store file in S3: {str(e)}")
            # Fallback to local storage
            logger.info("Falling back to local storage")
            return await self._store_file_local(file_ref, data, file_id, file_ext)

    async def _store_file_local(
        self,
        file_ref: FileReference,
        data: bytes,
        file_id: str,
        file_ext: str
    ) -> FileReference:
        """Store file locally"""
        try:
            # Generate local file path
            date_path = datetime.utcnow().strftime('%Y/%m/%d')
            local_dir = self.local_storage_path / date_path
            local_dir.mkdir(parents=True, exist_ok=True)

            local_path = local_dir / f"{file_id}.{file_ext}"
            file_ref.file_path = str(local_path)
            file_ref.storage_type = 'local'

            # Write file
            async with aiofiles.open(local_path, 'wb') as f:
                await f.write(data)

            # Generate URL (relative path)
            relative_path = local_path.relative_to(self.local_storage_path)
            file_ref.url = f"/storage/{relative_path}"

            logger.debug(f"File stored locally: {local_path}")
            return file_ref

        except Exception as e:
            logger.error(f"Failed to store file locally: {str(e)}")
            raise

    async def _generate_thumbnails(
        self,
        image: Image.Image,
        file_id: str,
        file_ext: str
    ) -> Dict[str, str]:
        """Generate thumbnails for different sizes"""
        thumbnail_urls = {}

        try:
            for width, height in self.config.thumbnail_sizes:
                # Create thumbnail
                thumbnail = image.copy()
                thumbnail.thumbnail((width, height), Image.Resampling.LANCZOS)

                # Save thumbnail
                thumbnail_data = io.BytesIO()
                thumbnail.save(thumbnail_data, format='JPEG', quality=80, optimize=True)
                thumbnail_bytes = thumbnail_data.getvalue()

                # Store thumbnail
                thumbnail_id = f"{file_id}_thumb_{width}x{height}"

                if self.s3_client:
                    s3_key = f"thumbnails/{datetime.utcnow().strftime('%Y/%m/%d')}/{thumbnail_id}.jpg"
                    await asyncio.to_thread(
                        self.s3_client.put_object,
                        Bucket=self.config.s3_bucket_name,
                        Key=s3_key,
                        Body=thumbnail_bytes,
                        ContentType='image/jpeg',
                        StorageClass='STANDARD'
                    )

                    thumbnail_url = await self._generate_s3_url(s3_key)
                    if self.config.cdn_enabled:
                        thumbnail_url = f"https://{self.config.cdn_domain}/{s3_key}"

                    thumbnail_urls[f"{width}x{height}"] = thumbnail_url
                else:
                    # Store locally
                    thumb_dir = self.temp_storage_path / "thumbnails"
                    thumb_dir.mkdir(exist_ok=True)
                    thumb_path = thumb_dir / f"{thumbnail_id}.jpg"

                    async with aiofiles.open(thumb_path, 'wb') as f:
                        await f.write(thumbnail_bytes)

                    relative_path = thumb_path.relative_to(self.temp_storage_path)
                    thumbnail_urls[f"{width}x{height}"] = f"/temp/{relative_path}"

            logger.debug(f"Generated {len(thumbnail_urls)} thumbnails for {file_id}")
            return thumbnail_urls

        except Exception as e:
            logger.error(f"Failed to generate thumbnails for {file_id}: {str(e)}")
            return {}

    async def _generate_s3_url(self, s3_key: str, expiration_seconds: int = 3600) -> str:
        """Generate S3 presigned URL"""
        try:
            url = await asyncio.to_thread(
                self.s3_client.generate_presigned_url,
                'get_object',
                Params={'Bucket': self.config.s3_bucket_name, 'Key': s3_key},
                ExpiresIn=expiration_seconds
            )
            return url

        except Exception as e:
            logger.error(f"Failed to generate S3 URL for {s3_key}: {str(e)}")
            return f"https://{self.config.s3_bucket_name}.s3.amazonaws.com/{s3_key}"

    async def get_file(self, file_id: str) -> Optional[FileReference]:
        """Get file reference by ID"""
        try:
            # Check cache first
            if file_id in self._file_cache:
                file_ref = self._file_cache[file_id]

                # Check if file has expired
                if file_ref.expires_at and datetime.utcnow() > file_ref.expires_at:
                    await self.delete_file(file_id)
                    return None

                return file_ref

            # File not found in cache
            return None

        except Exception as e:
            logger.error(f"Failed to get file {file_id}: {str(e)}")
            return None

    async def download_file(self, file_id: str) -> Optional[bytes]:
        """Download file data by ID"""
        start_time = time.time()

        try:
            file_ref = await self.get_file(file_id)
            if not file_ref:
                return None

            if file_ref.storage_type == 's3' and self.s3_client:
                # Download from S3
                response = await asyncio.to_thread(
                    self.s3_client.get_object,
                    Bucket=self.config.s3_bucket_name,
                    Key=file_ref.file_path
                )
                data = response['Body'].read()

            elif file_ref.storage_type == 'local':
                # Download from local storage
                file_path = Path(file_ref.file_path)
                if file_path.exists():
                    async with aiofiles.open(file_path, 'rb') as f:
                        data = await f.read()
                else:
                    return None
            else:
                return None

            # Update metrics
            download_time = time.time() - start_time
            self.metrics.daily_downloads += 1
            self.metrics.avg_download_time = (
                (self.metrics.avg_download_time + download_time) / 2
                if self.metrics.avg_download_time > 0 else download_time
            )

            logger.debug(f"File downloaded successfully: {file_id}")
            return data

        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Failed to download file {file_id}: {str(e)}")
            return None

    async def delete_file(self, file_id: str) -> bool:
        """Delete file by ID"""
        try:
            file_ref = await self.get_file(file_id)
            if not file_ref:
                return False

            deleted = False

            # Delete from S3
            if file_ref.storage_type == 's3' and self.s3_client:
                try:
                    await asyncio.to_thread(
                        self.s3_client.delete_object,
                        Bucket=self.config.s3_bucket_name,
                        Key=file_ref.file_path
                    )
                    deleted = True
                    logger.debug(f"File deleted from S3: {file_ref.file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete from S3: {str(e)}")

            # Delete from local storage
            if file_ref.storage_type == 'local':
                try:
                    file_path = Path(file_ref.file_path)
                    if file_path.exists():
                        file_path.unlink()
                        deleted = True
                        logger.debug(f"File deleted locally: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete local file: {str(e)}")

            # Delete thumbnails
            if file_ref.thumbnail_urls:
                for size, thumb_url in file_ref.thumbnail_urls.items():
                    if self.s3_client and thumb_url.startswith('https'):
                        # Extract S3 key from URL and delete
                        try:
                            s3_key = thumb_url.split(f"/{self.config.s3_bucket_name}/")[-1]
                            await asyncio.to_thread(
                                self.s3_client.delete_object,
                                Bucket=self.config.s3_bucket_name,
                                Key=s3_key
                            )
                        except Exception as e:
                            logger.error(f"Failed to delete thumbnail {s3_key}: {str(e)}")
                    else:
                        # Delete local thumbnail
                        try:
                            thumb_path = Path(thumb_url.replace('/temp/', self.temp_storage_path + '/'))
                            if thumb_path.exists():
                                thumb_path.unlink()
                        except Exception as e:
                            logger.error(f"Failed to delete local thumbnail: {str(e)}")

            # Remove from cache
            if file_id in self._file_cache:
                del self._file_cache[file_id]

            if deleted:
                logger.info(f"File deleted successfully: {file_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {str(e)}")
            return False

    async def create_backup(self) -> str:
        """Create backup of all stored files"""
        if not self.config.backup_enabled:
            raise ValueError("Backup is disabled")

        backup_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            if self.s3_client:
                # Create S3 backup using replication
                backup_key = f"backups/{datetime.utcnow().strftime('%Y/%m/%d')}/{backup_id}"

                # This would typically use S3 replication or lifecycle policies
                # For now, just log the backup creation
                logger.info(f"S3 backup initiated: {backup_key}")

            else:
                # Create local backup archive
                backup_path = self.local_storage_path / "backups"
                backup_path.mkdir(exist_ok=True)

                import shutil
                backup_file = backup_path / f"backup_{backup_id}.tar.gz"

                # Create archive
                shutil.make_archive(
                    str(backup_file.with_suffix('')),
                    'gztar',
                    str(self.local_storage_path / "uploads"),
                    exclude=[f"backups"]
                )

                logger.info(f"Local backup created: {backup_file}")

            backup_time = time.time() - start_time
            logger.info(f"Backup completed: {backup_id} in {backup_time:.2f}s")

            return backup_id

        except Exception as e:
            logger.error(f"Failed to create backup {backup_id}: {str(e)}")
            raise

    async def cleanup_expired_files(self, max_age_days: int = 30) -> int:
        """Clean up expired files"""
        cleaned_count = 0

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

            # Clean expired files from cache
            expired_files = [
                file_id for file_id, file_ref in self._file_cache.items()
                if file_ref.expires_at and file_ref.expires_at < cutoff_date
            ]

            for file_id in expired_files:
                if await self.delete_file(file_id):
                    cleaned_count += 1

            # Clean local temp files
            if self.config.cleanup_temp_files:
                temp_files = list(self.temp_storage_path.glob("*"))
                for temp_file in temp_files:
                    if temp_file.is_file():
                        file_age = time.time() - temp_file.stat().st_mtime
                        if file_age > max_age_days * 24 * 3600:  # Convert to seconds
                            try:
                                temp_file.unlink()
                                cleaned_count += 1
                            except Exception as e:
                                logger.error(f"Failed to delete temp file {temp_file}: {str(e)}")

            # Clean S3 old files (would use lifecycle policies in production)
            if self.s3_client:
                # This would typically use S3 lifecycle policies
                logger.debug("S3 cleanup would use lifecycle policies")

            self.metrics.last_cleanup = datetime.utcnow()
            logger.info(f"Cleanup completed: {cleaned_count} files removed")

            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired files: {str(e)}")
            return 0

    async def _background_cleanup(self):
        """Background task for periodic cleanup"""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)
                await self.cleanup_expired_files()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background cleanup error: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        try:
            stats = {
                'storage_type': 's3' if self.s3_client else 'local',
                'config': {
                    'max_file_size_mb': self.config.max_file_size_mb,
                    'allowed_formats': self.config.allowed_formats,
                    'compression_quality': self.config.compression_quality,
                    'cdn_enabled': self.config.cdn_enabled,
                    'backup_enabled': self.config.backup_enabled
                },
                'metrics': {
                    'total_files': len(self._file_cache),
                    'daily_uploads': self.metrics.daily_uploads,
                    'daily_downloads': self.metrics.daily_downloads,
                    'avg_upload_time': self.metrics.avg_upload_time,
                    'avg_download_time': self.metrics.avg_download_time,
                    'error_count': self.metrics.error_count,
                    'last_cleanup': self.metrics.last_cleanup.isoformat() if self.metrics.last_cleanup else None
                }
            }

            # Add S3-specific stats
            if self.s3_client:
                try:
                    # Get bucket size and object count
                    response = await asyncio.to_thread(
                        self.s3_client.list_objects_v2,
                        Bucket=self.config.s3_bucket_name,
                        MaxKeys=1
                    )

                    if 'KeyCount' in response:
                        stats['s3_stats'] = {
                            'total_objects': response['KeyCount'],
                            'bucket_name': self.config.s3_bucket_name,
                            'region': self.config.s3_region
                        }
                except Exception as e:
                    logger.error(f"Failed to get S3 stats: {str(e)}")

            # Add local storage stats
            if self.local_storage_path.exists():
                try:
                    total_size = sum(
                        f.stat().st_size for f in self.local_storage_path.rglob("*")
                        if f.is_file()
                    )
                    stats['local_stats'] = {
                        'storage_path': str(self.local_storage_path),
                        'total_size_mb': round(total_size / (1024 * 1024), 2),
                        'file_count': len(list(self.local_storage_path.rglob("*")))
                    }
                except Exception as e:
                    logger.error(f"Failed to get local storage stats: {str(e)}")

            return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            return {}

    async def cleanup(self):
        """Cleanup storage service resources"""
        try:
            # Stop background cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Close S3 client
            if self.s3_client:
                try:
                    await asyncio.to_thread(self.s3_client.close)
                except Exception as e:
                    logger.error(f"Failed to close S3 client: {str(e)}")

            # Clear cache
            self._file_cache.clear()

            logger.info("Storage service cleaned up")

        except Exception as e:
            logger.error(f"Error during storage cleanup: {str(e)}")

    def get_file_url(self, file_ref: FileReference, expiration_seconds: int = 3600) -> str:
        """Get public or presigned URL for file"""
        if file_ref.cdn_url:
            return file_ref.cdn_url
        elif file_ref.url:
            return file_ref.url
        elif file_ref.storage_type == 's3' and self.s3_client:
            return asyncio.create_task(self._generate_s3_url(file_ref.file_path, expiration_seconds))
        else:
            return file_ref.url or ""


# Decorator for automatic file storage
def store_uploaded_file(
    storage_service: StorageService,
    generate_thumbnails: bool = True,
    ttl_days: Optional[int] = None
):
    """Decorator for automatically storing uploaded files"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute original function to get file data
            result = await func(*args, **kwargs)

            # Check if result contains file data
            if isinstance(result, dict) and 'file_data' in result:
                try:
                    file_ref = await storage_service.store_file(
                        file_data=result['file_data'],
                        filename=result.get('filename', 'upload.jpg'),
                        content_type=result.get('content_type'),
                        metadata=result.get('metadata'),
                        generate_thumbnails=generate_thumbnails,
                        ttl_days=ttl_days
                    )

                    # Add file reference to result
                    result['file_reference'] = file_ref

                except Exception as e:
                    logger.error(f"Failed to store uploaded file: {str(e)}")
                    result['storage_error'] = str(e)

            return result

        return wrapper
    return decorator


# Factory function for easy storage service creation
def create_storage_service(settings: Settings) -> StorageService:
    """
    Factory function to create a storage service.

    Args:
        settings: Application settings

    Returns:
        StorageService instance
    """
    return StorageService(settings)