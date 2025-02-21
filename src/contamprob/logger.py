import logging
import logging.handlers
import sys


def init_logger(
    name, log_path=None, level_console=logging.WARNING, level_file=logging.WARNING
):
    """Initialize a logger.
    :param level_console: logging level for console logging output, defaulted to logging.WARNING
    :param level_file: logging level for log file logging output, defaulted to logging.WARNING
    :return :Nothing
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.error("Initialising already initialised logger!")
        return

    logger.info("Initialising logger.")
    logger.setLevel(level_console)

    formatter = logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.handlers[0].setLevel(level_console)

    if log_path is not None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.handlers[1].setLevel(level_file)


def close_logger(logger: logging.Logger):
    """Close a logger"""
    for h in logger.handlers:
        logger.removeHandler(h)
        h.flush()
        h.close()


class LoggingContext:
    """A context manager for logging."""

    def __init__(
        self,
        logger: logging.Logger,
        level: int | str,
        handler: logging.Handler | None = None,
    ):
        self.logger = logger
        self.level = level
        self.handler = handler

    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.level)
        if self.handler is not None:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        self.logger.setLevel(self.old_level)
        if self.handler is not None:
            self.logger.removeHandler(self.handler)
        if et is not None:
            self.logger.error(f"An exception occurred: {et}")
            self.logger.error(f"Exception value: {ev}")
            self.logger.error(f"Traceback: {tb}")


if __name__ == "__main__":
    init_logger(__name__, level_file=logging.CRITICAL, level_console=logging.DEBUG)
    log = logging.getLogger(__name__)
    log.debug("A DEBUG message is shown like this")
    log.info("Here you can see an INFO message")
    log.warning("This is a WARNING message")
    log.error("And this is an ERROR message")
    log.critical("Be careful if you see a CRITICAL error like this")