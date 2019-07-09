from abc import ABC, abstractmethod, ABCMeta

from pypadre.util.file_util import dir_list


class ISearchableFile:

    def _list(self):
        return dir_list(self.root_dir, None)
