import json
import logging
import logging.config
import sys
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "logging_config.json")

def setup_logging():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
            logger = logging.getLogger('cat_app')
            logger.debug(f"Логирование настроено из файла: {CONFIG_PATH}")
            return logger
        except Exception as e:
            print(f"Ошибка загрузки конфига: {e}. Использую резервную настройку.")

    return setup_basic_logging()

def setup_basic_logging():
    logger = logging.getLogger('cat_app')
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.log")

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')

    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.warning("Используется базовая конфигурация логирования")
    return logger

logger = None