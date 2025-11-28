"""
Logging Configuration

This module sets up structured logging for the application with JSON formatting
and appropriate log levels for different environments.
"""

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import structlog
from pythonjsonlogger import jsonlogger


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output"""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


class JSONFormatter(jsonlogger.JsonFormatter):
    """Enhanced JSON formatter with additional fields"""

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        """Add custom fields to log record"""
        super().add_fields(log_record, record, message_dict)

        # Add timestamp if not present
        if 'timestamp' not in log_record:
            log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'

        # Add log level
        if 'level' not in log_record:
            log_record['level'] = record.levelname.lower()

        # Add logger name
        if 'logger' not in log_record:
            log_record['logger'] = record.name

        # Add module and function if available
        if hasattr(record, 'module') and 'module' not in log_record:
            log_record['module'] = record.module
        if hasattr(record, 'funcName') and 'function' not in log_record:
            log_record['function'] = record.funcName

        # Add line number if available
        if hasattr(record, 'lineno') and 'line' not in log_record:
            log_record['line'] = record.lineno

        # Add process and thread info
        if 'process_id' not in log_record:
            log_record['process_id'] = record.process
        if 'thread_id' not in log_record:
            log_record['thread_id'] = record.thread

        # Add application context if available
        if hasattr(record, 'app_context'):
            log_record['app_context'] = record.app_context


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Setup structured logging for the application

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'text')
        log_file: Optional log file path
        console_output: Whether to output to console
    """

    # Validate log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Setup formatters
    if log_format.lower() == "json":
        console_formatter = JSONFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_formatter = JSONFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler to prevent huge log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Configure third-party loggers to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Setup structlog for better structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if log_format.lower() == "json" else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance

    Args:
        name: Logger name (usually __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class LoggerAdapter:
    """Adapter for adding context to log messages"""

    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.extra = extra or {}

    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log message with added context"""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.extra)
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)


def create_context_logger(
    name: str,
    **context: Any
) -> LoggerAdapter:
    """
    Create a logger with context bound to it

    Args:
        name: Logger name
        **context: Context to bind to logger

    Returns:
        Logger adapter with bound context
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)


class APILogger:
    """Specialized logger for API request/response logging"""

    def __init__(self, logger_name: str = "api"):
        self.logger = get_logger(logger_name)
        self.start_time = time.time()

    def log_request(self, method: str, url: str, headers: Dict[str, str], client_ip: str, **kwargs):
        """Log incoming API request"""
        self.logger.info(
            "api_request",
            method=method,
            url=url,
            client_ip=client_ip,
            user_agent=headers.get("user-agent", "unknown"),
            request_id=kwargs.get("request_id"),
            **{k: v for k, v in kwargs.items() if k != "request_id"}
        )

    def log_response(
        self,
        status_code: int,
        response_time_ms: int,
        response_size: int,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """Log API response"""
        level = logging.INFO if status_code < 400 else logging.ERROR

        self.logger.log(
            level,
            "api_response",
            status_code=status_code,
            response_time_ms=response_time_ms,
            response_size=response_size,
            request_id=request_id,
            success=status_code < 400,
            **kwargs
        )

    def log_error(self, error: Exception, request_id: Optional[str] = None, **kwargs):
        """Log API error"""
        self.logger.error(
            "api_error",
            error_type=type(error).__name__,
            error_message=str(error),
            request_id=request_id,
            **kwargs
        )

    def log_timing(self, operation: str, duration_ms: int, **kwargs):
        """Log operation timing"""
        level = logging.WARNING if duration_ms > 5000 else logging.INFO

        self.logger.log(
            level,
            "operation_timing",
            operation=operation,
            duration_ms=duration_ms,
            slow=duration_ms > 5000,
            **kwargs
        )


class AILogger:
    """Specialized logger for AI model operations"""

    def __init__(self, logger_name: str = "ai"):
        self.logger = get_logger(logger_name)

    def log_model_call(
        self,
        model_type: str,
        operation: str,
        processing_time_ms: int,
        success: bool,
        **kwargs
    ):
        """Log AI model operation"""
        level = logging.INFO if success else logging.WARNING

        self.logger.log(
            level,
            "ai_model_call",
            model_type=model_type,
            operation=operation,
            processing_time_ms=processing_time_ms,
            success=success,
            **kwargs
        )

    def log_model_error(self, model_type: str, error: Exception, **kwargs):
        """Log AI model error"""
        self.logger.error(
            "ai_model_error",
            model_type=model_type,
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs
        )

    def log_aggregation(
        self,
        models_used: list,
        primary_plant: str,
        primary_disease: str,
        plant_confidence: float,
        disease_confidence: float,
        models_agreed: bool,
        **kwargs
    ):
        """Log result aggregation"""
        self.logger.info(
            "ai_result_aggregation",
            models_used=models_used,
            primary_plant=primary_plant,
            primary_disease=primary_disease,
            plant_confidence=plant_confidence,
            disease_confidence=disease_confidence,
            models_agreed=models_agreed,
            **kwargs
        )


class DatabaseLogger:
    """Specialized logger for database operations"""

    def __init__(self, logger_name: str = "database"):
        self.logger = get_logger(logger_name)

    def log_query(self, query_type: str, table: str, duration_ms: int, **kwargs):
        """Log database query"""
        level = logging.WARNING if duration_ms > 1000 else logging.INFO

        self.logger.log(
            level,
            "database_query",
            query_type=query_type,
            table=table,
            duration_ms=duration_ms,
            slow_query=duration_ms > 1000,
            **kwargs
        )

    def log_connection(self, status: str, connection_type: str, **kwargs):
        """Log database connection"""
        self.logger.info(
            "database_connection",
            status=status,
            connection_type=connection_type,
            **kwargs
        )

    def log_cache_operation(self, operation: str, cache_type: str, hit: bool, **kwargs):
        """Log cache operation"""
        self.logger.info(
            "cache_operation",
            operation=operation,
            cache_type=cache_type,
            hit=hit,
            **kwargs
        )


# Convenience functions for creating specialized loggers
def get_api_logger(**context) -> APILogger:
    """Create API logger with optional context"""
    return APILogger()

def get_ai_logger(**context) -> AILogger:
    """Create AI logger with optional context"""
    return AILogger()

def get_database_logger(**context) -> DatabaseLogger:
    """Create database logger with optional context"""
    return DatabaseLogger()


# Example usage:
#
# Basic setup:
# from backend.utils.logging import setup_logging, get_logger
#
# setup_logging(
#     log_level="INFO",
#     log_format="json",
#     log_file="logs/app.log"
# )
#
# logger = get_logger(__name__)
# logger.info("Application started", extra={"version": "1.0.0"})
#
# # With context:
# from backend.utils.logging import create_context_logger
#
# context_logger = create_context_logger(__name__, user_id="123", session_id="abc")
# context_logger.info("User action performed", action="login")
#
# # Specialized loggers:
# from backend.utils.logging import get_api_logger, get_ai_logger
#
# api_logger = get_api_logger()
# api_logger.log_request("GET", "/api/v1/health", {}, "127.0.0.1")
#
# ai_logger = get_ai_logger()
# ai_logger.log_model_call("grok", "image_analysis", 1500, True, confidence=0.95)