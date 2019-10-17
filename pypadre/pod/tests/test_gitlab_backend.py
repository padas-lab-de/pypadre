import os
import shutil
import unittest

from pypadre.pod.app import PadreConfig
from pypadre.pod.app.padre_app import PadreAppFactory

config_path = os.path.join(os.path.expanduser("~"), ".padre_git_test.cfg")
workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-test")

class GitlabBackend(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(GitlabBackend, self).__init__(*args, **kwargs)
        config = PadreConfig(config_file=config_path)
        config.set("backends", str([
                    {
                        "root_dir": workspace_path,
                        "gitlab_url": 'http://localhost:8080/',
                        "user" : "root",
                        "token" : "e8mbtk4suozvmPw5c5Fo"
                    }
                ]))
        self.app = PadreAppFactory.get(config)

    def tearDown(self):
        # delete data content
        try:
            shutil.rmtree(workspace_path+"/datasets")
            shutil.rmtree(workspace_path+"/projects")
        except FileNotFoundError:
            pass

    @classmethod
    def tearDownClass(cls):
        # delete configuration
        if os.path.isdir(workspace_path):
            shutil.rmtree(workspace_path)
        os.remove(config_path)


    def test_gitlab_connection(self):

        backends = self.app.backends
        import gitlab

        server = gitlab.Gitlab(url='http://localhost:8080/',private_token="e8mbtk4suozvmPw5c5Fo")


        projects = server.projects.list()

        assert projects is not None
        # gitlab = GitLabRepository(root_dir=workspace_path,backend=)

        print("mehdi")

    def test_project(self):
        from pypadre.core.model.project import Project

        project = Project(name='Test Project', description='Testing the functionalities of project backend')
        self.app.projects.put(project)
        name = 'Test Project'
        projects = self.app.projects.list({'name': name})
        for project in projects:
            assert name in project.name

    def test_gitlab_backend(self):
        pass