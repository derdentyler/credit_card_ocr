import logging
import sys
from pathlib import Path

LOG_FILE_NAME = "project.log"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log_path = LOG_DIR / LOG_FILE_NAME

# Настройка форматирования
formatter = logging.Formatter(
    "%(asctime)s — %(levelname)s — %(name)s — %(message)s"
)

# Файловый хендлер
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)

# Консольный хендлер
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

# Основной логгер
logger = logging.getLogger("credit_card_ocr")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.propagate = False
