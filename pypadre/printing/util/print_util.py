# noinspection PyUnresolvedReferences
from StringIO import StringIO

from beautifultable import BeautifulTable
from beautifultable.enums import Alignment

import sys
import time
import threading

from pypadre.eventhandler import assert_condition


class StringBuilder:
    _file_str = None

    def __init__(self):
        self._file_str = StringIO()

    def append(self, str):
        self._file_str.write(str)

    def append_line(self, str):
        self._file_str.write(str+"\n")

    def __str__(self):
        return self._file_str.getvalue()


class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1:
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False


def to_table(table=None, *args):
    assert_condition(condition=all(isinstance(x, args[0].__class__) for x in args), source="to_table",
                     message="Arguments can't be printed in the same table. They are not homogeneous.")
    if table is None:
        table = get_default_table()
    table.column_headers = args[0].tablefy_header()
    with Spinner:
        for obj in args:
            table.append_row(
                [str(x) for x in obj.tablefy_to_row()])
    return table


def get_default_table():
    table = BeautifulTable(max_width=150, default_alignment=Alignment.ALIGN_LEFT)
    table.row_separator_char = ""
    return table


def print_dicts_as_table(dicts, separate_head=True, heads=None):
    """Prints a list of dicts as table"""

    def _convert(x):
        if x is None:
            return ""
        elif hasattr(x, "__len__") and not isinstance(x, str):
            return " + ".join([_convert(xi) for xi in x])
        else:
            return str(x)

    if heads is None:
        heads = set([k for d in dicts for k in d.keys()])
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
