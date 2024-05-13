import os
import logging

dirname = os.getcwd()
DEFAULT_LOGFILE = os.path.join(dirname, "out.log")


def logging_config(logfile: str = None):
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(filename=logfile, mode="w"), logging.StreamHandler()],
    )


def get_logger(name: str, logfile: str = None, console=True):
    # logging.basicConfig(level=logging.INFO, force=True)  ## override Jupyter notebook logger
    logger = logging.getLogger(name)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create file handler
    if logfile is not None:
        fh = logging.FileHandler(filename=logfile, encoding="utf-8")
        fh.setLevel(logging.INFO)

        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if console:
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    log_filter = logging.Filter(name)
    logger.addFilter(log_filter)

    return logger
