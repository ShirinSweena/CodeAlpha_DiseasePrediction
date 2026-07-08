import sys
from pathlib import Path
from loguru import logger

# Ensure output directory exists for logging
log_dir = Path("outputs/logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logger
logger.remove()  # Remove default handler

# Console handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# File handler
logger.add(
    log_dir / "pipeline.log",
    rotation="10 MB",
    retention="10 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)