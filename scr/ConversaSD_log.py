import logging
import os
from datetime import datetime

def setup_logger(log_level="INFO", log_file="log_execution.log"):
    # Crée le dossier "logs" dans le dossier du script s’il n’existe pas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Ajoute la date au fichier log
    dated_log_file = os.path.join(log_dir, f"{datetime.now():%Y%m%d_%H%M%S}_{log_file}")

    logger = logging.getLogger("ConversaSD")
    logger.setLevel(log_level.upper())
    logger.handlers = []  # Reset pour éviter les doublons

    # Format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Handler fichier
    file_handler = logging.FileHandler(dated_log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
