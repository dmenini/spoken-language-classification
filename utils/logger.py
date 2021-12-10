import logging
import os


def get_logger(path, name):
    filehandler = logging.FileHandler(os.path.join(path, '{}.logs'.format(name)), 'a')
    consolehandler = logging.StreamHandler()

    formatter = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    filehandler.setFormatter(formatter)
    consolehandler.setFormatter(formatter)

    logger = logging.getLogger(name)

    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)

    logger.addHandler(filehandler)  # set the new handler
    logger.addHandler(consolehandler)

    logger.setLevel(logging.DEBUG)

    return logger
