import logging

log_file = "inference.log"


def logging_config(output_file: str = None):
    logging.basicConfig(
        level=logging.DEBUG,
        force=True,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(filename=output_file, mode="w"), logging.StreamHandler()],
    )


def get_logger(name: str, output_file: str = None, console=True):
    # logging.basicConfig(level=logging.INFO, force=True)  ## override Jupyter notebook logger
    logger = logging.getLogger(name)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    # Create file handler
    if output_file is not None:
        fh = logging.FileHandler(filename=output_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)

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
