import random
import tempfile
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from .constants import DEFAULT_FORMAT, DEFAULT_APP_LOGPATH, RESOURCE_DIRECTORY_PATH, DEBUG

class DefaultLogger:
    @staticmethod
    def get_default_logger():
        # logging.basicConfig(filename=DEFAULT_APP_LOGPATH, level=logging.DEBUG, format=DEFAULT_FORMAT)
        if DEBUG:
            logging.basicConfig(level=logging.DEBUG, format=DEFAULT_FORMAT)
        else:
            logging.basicConfig(filename=DEFAULT_APP_LOGPATH, level=logging.DEBUG, format=DEFAULT_FORMAT)
        #logging.disabled = True  # 'True' for development purpose. Should be changed to 'False' before deploying
        return logging
    # TODO: configure the logger to just log application logs and not the internal server logs.
    # TODO: add methods to create custom logger with custom format
    # TODO: convert it to static method and always return same logger object

class ResourceDirectory:
    def create_directory(self):
        # TODO create a corresponding configuration object. look up best practices
        data_dir = os.path.expanduser(RESOURCE_DIRECTORY_PATH)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir
