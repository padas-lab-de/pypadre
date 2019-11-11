# from pypadre.pod.generic.i_repository import ISourceRepository
#
#
# class SourceFileRepository(ISourceRepository):
#
#     @staticmethod
#     def placeholder():
#         return '{SOURCE_PATH}'
#
#     NAME = 'souce_code'
#     FILE_LIMIT = 30
#     FOLDER_LIMIT = 10
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def to_folder_name(self, source):
#         return source.name
#
#     def get(self, uid):
#         """
#         Shortcut because we know the uid is the folder name
#         :param uid: Uid of the execution
#         :return:
#         """
#         # TODO might be changed. Execution get folder name or id by git commit hash?
#         return super().get(uid)
#
#     def get_by_dir(self, directory):
#         # Souce code will not get_by_dir
#         pass
#
#     def put(self, obj, *args, merge=False, allow_overwrite=False, **kwargs):
#         # Create a symbolic link in the source folder
#         import os
#         source = obj
#         directory = source.path
#         if not os.path.isdir(directory):
#             raise ValueError('Source code path has to be a directory')
#
#         if not self.repo_exists(directory):
#             # Check how many files are existing in the directory if git is not present
#             import os
#
#             files = folders = 0
#
#             for _, dirnames, filenames in os.walk(path):
#                 # ^ this idiom means "we won't be using this value"
#                 files += len(filenames)
#                 folders += len(dirnames)
#
#             if files > self.FILE_LIMIT or folders > self.FOLDER_LIMIT:
#                 print('Large number of files found in source directory')
#
#             else:
#                 repo = self._create_repo(directory, bare=False)
#                 self._add_untracked_files(repo=repo)
#
#         else:
#             repo = self.get_repo(path=directory)
#             self._add_untracked_files(repo=repo)
#             if self._has_uncommitted_files(repo=repo):
#                 self._commit(repo=repo, message='Files committed by PyPaDRe')
#
#         if directory is not None:
#
#             path = os.path.join(directory, source.experiment.name)
#             # Add a symbolic link to the experiment folder in the source directory if it doesn't exist
#             if not os.path.exists(path):
#                 os.symlink(directory, path)
#
#             # Add symlink to the .gitignore
#             with open(os.path.join(directory, '.git', '.gitignore'), "a") as f:
#                 f.write(''.join('\n*',source.experiment.name,'*'))
#
#
#
