import logging
import time
from pathlib import Path
from contextlib import contextmanager


def get_ablation_logger(name: str, log_dir: Path):
    """
    Creates a formatted logger for ablation studies.
    Logs both to a file and the console.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers during reruns
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


@contextmanager
def log_stage(logger, stage_name: str):
    """
    Context manager for starting and finishing a stage with timing.
    """
    logger.info(f"🔄 Starting: {stage_name}")
    start = time.time()
    yield
    end = time.time()
    logger.info(f"✅ Completed: {stage_name} in {end - start:.2f} seconds")
