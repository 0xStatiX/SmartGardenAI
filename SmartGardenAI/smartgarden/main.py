"""
SmartGardenAI - Main FastAPI Application
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import List, Optional
import asyncio

from smartgarden.core.config import settings
from smartgarden.core.database import init_db, get_db
from smartgarden.core.security import verify_token
from smartgarden.api.v1.router import api_router
from smartgarden.core.middleware import RequestLoggingMiddleware
from smartgarden.services.ml_service import MLService
from smartgarden.services.iot_service import IoTService
from smartgarden.services.notification_service import NotificationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting SmartGardenAI application...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized successfully")
    
    # Initialize services
    app.state.ml_service = MLService()
    app.state.iot_service = IoTService()
    app.state.notification_service = NotificationService()
    
    # Start background tasks
    asyncio.create_task(app.state.iot_service.start_monitoring())
    asyncio.create_task(app.state.ml_service.start_training_scheduler())
    
    logger.info("SmartGardenAI application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SmartGardenAI application...")
    
    # Stop background tasks
    await app.state.iot_service.stop_monitoring()
    await app.state.ml_service.stop_training_scheduler()
    
    logger.info("SmartGardenAI application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="SmartGardenAI",
    description="Intelligent IoT-based smart garden system with AI-powered insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)
app.add_middleware(RequestLoggingMiddleware)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "ml_service": "running",
            "iot_service": "running",
            "notification_service": "running"
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to SmartGardenAI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    try:
        user = await verify_token(credentials.credentials)
        return user
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Protected endpoint example
@app.get("/api/protected")
async def protected_endpoint(current_user = Depends(get_current_user)):
    """Example protected endpoint"""
    return {"message": f"Hello {current_user.username}!", "user_id": current_user.id}

# Background task example
@app.post("/api/background-task")
async def trigger_background_task(
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """Trigger a background task"""
    background_tasks.add_task(process_background_task)
    return {"message": "Background task triggered"}

async def process_background_task():
    """Example background task"""
    logger.info("Processing background task...")
    await asyncio.sleep(5)
    logger.info("Background task completed")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "path": request.url.path
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time garden updates"""
    await websocket.accept()
    try:
        while True:
            # Send real-time sensor data
            sensor_data = await get_latest_sensor_data()
            await websocket.send_json(sensor_data)
            await asyncio.sleep(5)  # Update every 5 seconds
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

async def get_latest_sensor_data():
    """Get latest sensor data for WebSocket"""
    # This would typically fetch from database or IoT service
    return {
        "temperature": 24.5,
        "humidity": 65.2,
        "soil_moisture": 0.7,
        "light_intensity": 850,
        "timestamp": "2024-01-15T10:30:00Z"
    }

# Metrics endpoint for monitoring
@app.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    return {
        "uptime": "2 hours 30 minutes",
        "requests_per_minute": 150,
        "active_connections": 25,
        "memory_usage": "45%",
        "cpu_usage": "12%"
    }

# Configuration endpoint
@app.get("/api/config")
async def get_config():
    """Get application configuration (non-sensitive)"""
    return {
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "allowed_hosts": settings.ALLOWED_HOSTS,
        "database_url": "***hidden***",
        "redis_url": "***hidden***"
    }

# System status endpoint
@app.get("/api/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    return {
        "application": {
            "status": "running",
            "version": "1.0.0",
            "uptime": "2 hours 30 minutes"
        },
        "database": {
            "status": "connected",
            "connections": 5,
            "response_time": "15ms"
        },
        "iot_gateway": {
            "status": "connected",
            "sensors_online": 8,
            "last_update": "2024-01-15T10:29:45Z"
        },
        "ml_models": {
            "plant_health_predictor": "loaded",
            "watering_optimizer": "loaded",
            "last_training": "2024-01-15T08:00:00Z"
        },
        "notifications": {
            "email": "enabled",
            "sms": "enabled",
            "push": "enabled"
        }
    }

# Maintenance endpoints
@app.post("/api/maintenance/backup")
async def trigger_backup(background_tasks: BackgroundTasks):
    """Trigger database backup"""
    background_tasks.add_task(perform_backup)
    return {"message": "Backup initiated"}

async def perform_backup():
    """Perform database backup"""
    logger.info("Starting database backup...")
    # Backup logic here
    await asyncio.sleep(10)
    logger.info("Database backup completed")

@app.post("/api/maintenance/cleanup")
async def trigger_cleanup(background_tasks: BackgroundTasks):
    """Trigger data cleanup"""
    background_tasks.add_task(perform_cleanup)
    return {"message": "Cleanup initiated"}

async def perform_cleanup():
    """Perform data cleanup"""
    logger.info("Starting data cleanup...")
    # Cleanup logic here
    await asyncio.sleep(5)
    logger.info("Data cleanup completed")

# Development endpoints (only in debug mode)
if settings.DEBUG:
    @app.get("/api/debug/info")
    async def debug_info():
        """Debug information endpoint"""
        return {
            "debug": True,
            "environment": settings.ENVIRONMENT,
            "database_url": settings.DATABASE_URL,
            "redis_url": settings.REDIS_URL,
            "allowed_hosts": settings.ALLOWED_HOSTS
        }

if __name__ == "__main__":
    uvicorn.run(
        "smartgarden.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    ) 