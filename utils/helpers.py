"""
helpers.py - Utility functions for the Fake Review Detection System.

Provides logging, timing, data I/O, validation, and common transformations
used across all modules in the pipeline.
"""

import os
import time
import json
import logging
import hashlib
import functools
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from colorama import Fore, Style, init

init(autoreset=True)

# ── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"

for _dir in [DATA_DIR, LOG_DIR, MODEL_DIR, CACHE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# ── Logger Setup ────────────────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a named logger with console + file handlers.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler with color
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_file = LOG_DIR / f"{name.replace('.', '_')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# ── Decorators ──────────────────────────────────────────────────────────────
def timer(func):
    """Decorator that logs execution time of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(
            f"{Fore.CYAN}⏱  {func.__name__} "
            f"completed in {elapsed:.3f}s{Style.RESET_ALL}"
        )
        return result

    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Initial delay between retries (seconds).
        backoff: Multiplier applied to delay after each retry.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(
                        f"{Fore.YELLOW}⚠  Attempt {attempt}/{max_attempts} "
                        f"failed for {func.__name__}: {e}. "
                        f"Retrying in {_delay:.1f}s...{Style.RESET_ALL}"
                    )
                    time.sleep(_delay)
                    _delay *= backoff

        return wrapper

    return decorator


# ── Data I/O ────────────────────────────────────────────────────────────────
def load_csv(filename: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load a CSV file from the data directory.

    Args:
        filename: Name of the CSV file.
        data_dir: Optional override for the data directory path.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = (data_dir or DATA_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    print(
        f"{Fore.GREEN}✓  Loaded {filename}: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns{Style.RESET_ALL}"
    )
    return df


def save_csv(df: pd.DataFrame, filename: str, data_dir: Optional[Path] = None):
    """Save a DataFrame to CSV in the data directory."""
    path = (data_dir or DATA_DIR) / filename
    df.to_csv(path, index=False, encoding="utf-8")
    print(
        f"{Fore.GREEN}✓  Saved {filename}: "
        f"{df.shape[0]:,} rows × {df.shape[1]} columns{Style.RESET_ALL}"
    )


def save_json(data: Any, filename: str, directory: Optional[Path] = None):
    """Save data as JSON."""
    path = (directory or DATA_DIR) / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"{Fore.GREEN}✓  Saved JSON: {filename}{Style.RESET_ALL}")


def load_json(filename: str, directory: Optional[Path] = None) -> Any:
    """Load data from a JSON file."""
    path = (directory or DATA_DIR) / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Validation ──────────────────────────────────────────────────────────────
def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame",
) -> bool:
    """
    Validate that a DataFrame has the required columns.

    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must exist.
        name: Descriptive name for error messages.

    Returns:
        True if valid.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return True


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a data quality summary for a DataFrame.

    Returns:
        Dictionary with quality metrics including null counts,
        duplicates, and dtype distribution.
    """
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }


# ── Text Utilities ──────────────────────────────────────────────────────────
def hash_text(text: str) -> str:
    """Generate an MD5 hash of a text string for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length, appending ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


# ── Numeric Utilities ───────────────────────────────────────────────────────
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Perform division, returning a default value if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1] range."""
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


# ── Pretty Printing ────────────────────────────────────────────────────────
def print_header(title: str, width: int = 60):
    """Print a formatted section header."""
    print(f"\n{Fore.MAGENTA}{'═' * width}")
    print(f"  {title.upper()}")
    print(f"{'═' * width}{Style.RESET_ALL}\n")


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Print a dictionary of metrics in a clean table format."""
    print_header(title)
    max_key_len = max(len(k) for k in metrics)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {Fore.WHITE}{key:<{max_key_len}} : {Fore.CYAN}{value:.4f}")
        else:
            print(f"  {Fore.WHITE}{key:<{max_key_len}} : {Fore.CYAN}{value}")
    print()


def print_progress(current: int, total: int, prefix: str = "Progress"):
    """Print a simple progress indicator."""
    pct = current / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(
        f"\r  {prefix}: [{Fore.CYAN}{bar}{Style.RESET_ALL}] "
        f"{pct:5.1f}% ({current}/{total})",
        end="",
        flush=True,
    )
    if current == total:
        print()
