import logging
import os
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join("bop_toolkit")))


log_dir = Path("logs/")
if not log_dir.is_dir():
    log_dir.mkdir(parents=True)
logging_format = logging.Formatter(
    "%(asctime)s,%(msecs)d %(name)s | %(levelname)s | %(message)s"
)


def get_logger(name) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(log_dir / "all_logs.log")
        handler.setFormatter(logging_format)
        logger.addHandler(handler)
        stdouthandler = logging.StreamHandler(sys.stdout)
        stdouthandler.setFormatter(logging_format)
        logger.addHandler(stdouthandler)
    return logger


def add_file_handler(logger: logging.Logger, path: Path):
    """
    Make a logger write to an additional file path
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging_format)
    logger.addHandler(handler)


logger = get_logger(__name__)


# write unhandled exceptions to a log file in addition to whatever python does
def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    # call default excepthook
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    if issubclass(exc_type, KeyboardInterrupt):
        return
    # create a critical level log message with info from the except hook.
    logger.critical(
        "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


# override sys excepthook to do the usual exception callback but also
# log the exception.
sys.excepthook = handle_unhandled_exception
