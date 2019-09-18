# import os
#
# from pypadre.pod.backend.local.file.dataset.dataset_file_backend import DatasetFileRepository
#
#
# class PadreDatasetHttpRepository(DatasetFileRepository):
#
#     def __init__(self, parent):
#         super().__init__(parent)
#         self.root_dir = os.path.join(self._parent.root_dir, "datasets")
#
#     def put_progress(self, obj):
#         pass
#
#     def list(self, search, offset=0, size=100):
#         """
#         List all data sets in the repository
#         :param offset:
#         :param size:
#         :param **args:
#         :param search_name: regular expression based search string for the title. Default None
#         :param search_metadata: dict with regular expressions per metadata key. Default None
#         """
#         # todo apply the search metadata filter.
#         return super().list(search, offset=offset, size=size)
#
#     def get(self, uid):
#         pass
#
#     def put(self, obj, *args, merge=False, **kwargs):
#         pass
#
#     def delete(self, uid):
#         pass