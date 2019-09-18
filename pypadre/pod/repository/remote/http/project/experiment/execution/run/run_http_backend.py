# import os
#
# from pypadre.pod.backend.local.file.project.experiment.execution.run.run_file_backend import RunFileRepository
# from pypadre.pod.backend.remote.http.project.experiment.execution.run.split.split_http_backend import \
#     PadreSplitHttpRepository
#
#
# class PadreRunHttpRepository(RunFileRepository):
#
#     def __init__(self, parent):
#         super().__init__(parent)
#         self.root_dir = os.path.join(self._parent.root_dir, "runs")
#         self._split = PadreSplitHttpRepository(self)
#
#     @property
#     def split(self):
#         return self._split
#
#     def put_progress(self, obj):
#         pass
#
#     def list(self, search, offset=0, size=100):
#         pass
#
#     def get(self, uid):
#         pass
#
#     def put(self, obj, *args, merge=False, **kwargs):
#         pass
#
#     def delete(self, uid):
#         pass