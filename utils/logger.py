"""
Logger utility for logging messages
Prints logs to screen via ch (channel handler), and saves logs to via fh (file handler)
"""
import logging

# Logger config
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)

logger.addHandler(ch)
