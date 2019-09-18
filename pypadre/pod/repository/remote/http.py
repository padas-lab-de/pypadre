# from pypadre.pod.backend.interfaces.backend.generic.i_basic_mixins import IRepository
# from pypadre.pod.backend.interfaces.backend.i_padre_backends import IDatasetRepository, IExperimentRepository, IRunRepository, \
#     ISplitRepository
#
#
# class DatasetFileRepository(IDatasetRepository):
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
#
#
# class ExperimentFileRepository(IExperimentRepository):
#
#     def put_config(self, obj):
#         pass
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
#
#
# class RunFileRepository(IRunRepository):
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
#
#
# class SplitFileRepository(ISplitRepository):
#
#     def put_progress(self, obj):pass
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
#
#
# class PadreFileRepository(IRepository):
#     """
#     Delegator class for handling padre objects at the file repository level. The following files tructure is used:
#
#     root_dir
#       |------datasets\
#       |------experiments\
#     """
#
#     @property
#     def dataset(self):
#         pass
#
#     @property
#     def project(self):
#         pass
#
#     @property
#     def experiment(self):
#         pass
#
#     @property
#     def execution(self):
#         pass
#
#     @property
#     def result(self):
#         pass
#
#     @property
#     def run(self):
#         pass
#
#     @property
#     def split(self):
#         pass
