"""
Middleware and exception handlers for the Plant Disease Detection API
"""

import logging
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


def setup_logging():
    """Setup logging configuration"""
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"

    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('api.log')
        ]
    )


def setup_exception_handlers(app: FastAPI):
    """Setup global exception handlers"""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger = logging.getLogger(__name__)
        logger.warning(f"HTTP {exc.status_code}: {exc.detail} - Path: {request.url.path}")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": request.url.path
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger = logging.getLogger(__name__)
        logger.warning(f"Validation error: {exc.errors()} - Path: {request.url.path}")

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Validation error",
                "details": exc.errors(),
                "path": request.url.path
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {str(exc)} - Path: {request.url.path}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "path": request.url.path
            }
        )