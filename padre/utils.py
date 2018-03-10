import random
import tempfile
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from .constants import DEFAULT_FORMAT, DEFAULT_APP_LOGPATH, RESOURCE_DIRECTORY_PATH, DEBUG
import ctypes

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


FILE_ATTRIBUTE_HIDDEN = 0x02

def write_hidden(file_name, data):
    """
    Cross platform hidden file writer.
    see https://stackoverflow.com/questions/25432139/python-cross-platform-hidden-file
    """
    # For *nix add a '.' prefix.
    prefix = '.' if os.name != 'nt' else ''
    file_name = prefix + file_name

    # Write file.
    with open(file_name, 'w') as f:
        f.write(data)

    # For windows set file attribute.
    if os.name == 'nt':
        ret = ctypes.windll.kernel32.SetFileAttributesW(file_name,
                                                        FILE_ATTRIBUTE_HIDDEN)
        if not ret: # There was an error.
            raise ctypes.WinError()


def print_dicts_as_table(dicts, separate_head=True, heads = None):
    """Prints a list of dicts as table"""
    def _convert(x):
        if x is None:
            return ""
        elif hasattr(x, "__len__") and not isinstance(x,str):
            return " + ".join([_convert(xi) for xi in x])
        else:
            return str(x)
    if heads is None:
        heads = set([ k  for d in dicts for k in d.keys()])
    table = [[k for k in heads]]
    for d in dicts:
        table.append([_convert(d[k]) for k in heads])
    print_table(table, separate_head)


def print_table(lines, separate_head=True):
    """Prints a formatted table given a 2 dimensional array"""
    # Count the column width
    widths = []
    for line in lines:
        for i, x in enumerate(line):
            if x is None:
                size = 0
            else:
                size = len(x)

            while i >= len(widths):
                widths.append(0)
            if size > widths[i]:
                widths[i] = size

    # Generate the format string to pad the columns
    print_string = ""
    for i, width in enumerate(widths):
        print_string += "{" + str(i) + ":" + str(width) + "} | "
    if (len(print_string) == 0):
        return
    print_string = print_string[:-3]

    # Print the actual data
    for i, line in enumerate(lines):
        print(print_string.format(*line))
        if (i == 0 and separate_head):
            print("-" * (sum(widths) + 3 * (len(widths) - 1)))


class _const:

    class ConstError(TypeError): pass

    def __setattr__(self,name,value):
        if self.__dict__.has_key(name):
            raise self.ConstError("Can't rebind const(%s)" % name)
        self.__dict__[name]=value