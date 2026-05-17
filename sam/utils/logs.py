#!/usr/bin/env python
# encoding: utf-8
# Copyright (c) 2025- SATURN2
# AUTHORS
# Sukbong Kwon (Galois)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d: %(name)-10.10s: %(levelname)s: %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d %p %I:%M:%S",
)

def get_logger(
    name: str,
    level: str = 'INFO'
)-> logging.Logger:
    """Get name for logger from current module

    Args:
        name (str): Name of module
        level (int): Logging level
            logging.DEBUG
            logging.INFO
            logging.WARNING
            logging.ERROR
            logging.CRITICAL
            logging.NOTSET

    Returns:
        logging.Logger: Logger object
    """
    logger = logging.getLogger(name.split('.')[-1])

    # Set logging level from string
    level = level.upper()
    if hasattr(logging, level):
        logger.setLevel(getattr(logging, level))
    else:
        logger.setLevel(logging.INFO)

    return logger
