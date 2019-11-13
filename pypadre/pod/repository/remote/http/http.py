# from pypadre.pod.backend.local.file.file import FileRepository
# from pypadre.pod.backend.remote.http.dataset.dataset_http_backend import DatasetHttpRepository
# from pypadre.pod.backend.remote.http.project.project_http_backend import ProjectHttpRepository
#
#
# class PadreHttpRepository(PadreFileRepository):
#     """
#     Delegator class for handling padre objects at the file generic level. The following files tructure is used:
#
#     root_dir
#       |------datasets\
#       |------experiments\
#     """
#
#     def __init__(self, config):
#         # TODO defensive programing: add check for root_dir
#         super().__init__(config)
#         self._dataset = PadreDatasetHttpRepository(self)
#         self._experiment_repository = PadreProjectHttpRepository(self)
