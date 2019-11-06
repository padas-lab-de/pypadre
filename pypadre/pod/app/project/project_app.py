from typing import List

from pypadre.pod.app.base_app import BaseChildApp
from pypadre.pod.repository.i_repository import IProjectRepository
from pypadre.pod.service.project_service import ProjectService


class ProjectApp(BaseChildApp):

    def __init__(self, parent, backends: List[IProjectRepository], **kwargs):
        super().__init__(parent, service=ProjectService(backends=backends), **kwargs)

    # class Decorators:
    #     def __init__(self, app, project):
    #         self._app = app
    #         self._project = project
    #
    #     def experiment(self, *args, **kwargs):
    #         def experiment_decorator(f_experiment):
    #             model = experiment_model(*args, project=self._project, **kwargs)
    #             @wraps(f_experiment)
    #             def wrap_experiment(*args, **kwargs):
    #                 return wrap_experiment
    #             Experiment.from_model(model)
    #         return experiment_decorator

    def execute(self, id):
        self.service.execute(id)

    def create(self, *args, **kwargs):
        """
        Function creates a project and puts the project to the backend
        :return:
        """
        project = self.service.create(*args, **kwargs)
        self.put(project)

        # Add decorator functions
        # self._add_decorators(project)
        return project

    def list(self, search=None, offset=0, size=100) -> list:
        obj_list = super().list(search, offset, size)

        # Add decorator functions
        # map(self._add_decorators, obj_list)
        return obj_list

    def get_by_name(self, name):
        found = self.service.list({"name": name})
        if found:
            return found.pop()
        return None

    # def get(self, id):
    #
    #     # Add decorator functions
    #     return self._add_decorators(super().get(id))

    # def _add_decorators(self, obj):
    #     return super()._add_clz_decorators(self.Decorators, obj)
