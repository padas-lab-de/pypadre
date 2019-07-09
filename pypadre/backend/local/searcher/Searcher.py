class PadreFileSearcher(ISearchable):

    def __init__(self, parent):
        self._parent = parent;

    def find(self, search):
        # TODO faster bash methods for linux etc
        _dir_list(self.root_dir, "")