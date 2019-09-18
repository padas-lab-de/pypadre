# import os
#
# from pypadre.pod.backend.local.file.project.experiment.execution.execution_file_backend import \
#     ExecutionFileRepository
# from pypadre.pod.backend.remote.http.project.experiment.execution.run.run_http_backend import RunHttpRepository
#
#
# class PadreExecutionHttpRepository(ExecutionFileRepository):
#
#     def __init__(self, parent):
#         super().__init__(parent)
#         self.root_dir = os.path.join(self._parent.root_dir, "executions")
#         self._run = PadreRunHttpRepository(self)
#
#     @property
#     def run(self):
#         return self._run
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