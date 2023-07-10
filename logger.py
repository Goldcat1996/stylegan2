import logging
from logging import StreamHandler


def mylogger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


logger = mylogger(__name__)


