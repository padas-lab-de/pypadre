from pypadre.backend.interfaces.backend.i_result_backend import IResultBackend


class PadreResultHTTPBackend(IResultBackend):

    def __init__(self, parent):
        super().__init__(parent)

    def list(self, search):
        pass

    def get(self, uid):
        pass

    def put(self, obj):
        pass

    def delete(self, uid):
        pass