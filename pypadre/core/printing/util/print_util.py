# noinspection PyUnresolvedReferences
import itertools
import sys
import threading
import time
from io import StringIO
from typing import List, Union

from beautifultable import BeautifulTable
from beautifultable.enums import Alignment, WidthExceedPolicy

from pypadre.core.printing.tablefyable import Tablefyable


class StringBuilder:
    _file_str = None

    def __init__(self):
        self._file_str = StringIO()

    def append(self, str):
        self._file_str.write(str)

    def append_line(self, str):
        self._file_str.write(str + "\n")

    def __str__(self):
        return self._file_str.getvalue()


class Spinner(object):
    spinner_cycle = itertools.cycle(['-', '/', '|', '\\'])

    def __init__(self):
        self.stop_running = threading.Event()
        self.spin_thread = threading.Thread(target=self.init_spin)

    def start(self):
        self.spin_thread.start()

    def stop(self):
        self.stop_running.set()
        self.spin_thread.join()

    def init_spin(self):
        while not self.stop_running.is_set():
            sys.stdout.write(next(self.spinner_cycle))
            sys.stdout.flush()
            time.sleep(0.25)
            sys.stdout.write('\r')


def get_default_table():
    table = BeautifulTable(max_width=250, default_alignment=Alignment.ALIGN_LEFT)
    table.row_separator_char = ""
    return table


def to_table(clz, objects: Union[Tablefyable, List[Tablefyable]], columns=None, table=None, spinner=False,
             print_empty=True):
    if clz is None:
        if objects:
            if not isinstance(objects, List):
                objects = [objects]
            clz = objects[0].__class__
    if table is None:
        table = get_default_table()
    if columns is None:
        table.column_headers = clz.tablefy_header()
    else:
        table.column_headers = clz.tablefy_header(*columns)
    if spinner:
        spinner = Spinner()
        spinner.start()
    for obj in objects:
        table.append_row(
            [str(x) for x in obj.tablefy_to_row(*table.column_headers)])
    if spinner:
        spinner.stop()
    if print_empty and len(table) == 0:
        table.append_row([str("-") for x in table.column_headers])
    return table

# def print_dicts_as_table(dicts, separate_head=True, heads=None):
#     """Prints a list of dicts as table"""
#
#     def _convert(x):
#         if x is None:
#             return ""
#         elif hasattr(x, "__len__") and not isinstance(x, str):
#             return " + ".join([_convert(xi) for xi in x])
#         else:
#             return str(x)
#
#     if heads is None:
#         heads = set([k for d in dicts for k in d.keys()])
#     table = [[k for k in heads]]
#     for d in dicts:
#         table.append([_convert(d[k]) for k in heads])
#     print_table(table, separate_head)
#
#
# def print_table(lines, separate_head=True):
#     """Prints a formatted table given a 2 dimensional array"""
#     # Count the column width
#     widths = []
#     for line in lines:
#         for i, x in enumerate(line):
#             if x is None:
#                 size = 0
#             else:
#                 size = len(x)
#
#             while i >= len(widths):
#                 widths.append(0)
#             if size > widths[i]:
#                 widths[i] = size
#
#     # Generate the format string to pad the columns
#     print_string = ""
#     for i, width in enumerate(widths):
#         print_string += "{" + str(i) + ":" + str(width) + "} | "
#     if (len(print_string) == 0):
#         return
#     print_string = print_string[:-3]
#
#     # Print the actual data
#     for i, line in enumerate(lines):
#         print(print_string.format(*line))
#         if (i == 0 and separate_head):
#             print("-" * (sum(widths) + 3 * (len(widths) - 1)))
