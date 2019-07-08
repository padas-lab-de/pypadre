class ProjectApp:
    def __init__(self, parent):
        self._parent = parent

    def delete(self, search):
        """
        Deletes the projects given in :param search: str to search experiment name or dict object with format
        {field : regexp<String>} pattern to search in particular fields using a regexp.
        """