import logging
import sys

def setup_logging():

    logger = logging.getLogger('cat_app')
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler('app.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')

    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()