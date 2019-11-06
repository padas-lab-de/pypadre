import unittest

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.model.code.icode import Function
from pypadre.pod.tests.base_test import PadreAppTest


class AppLocalBackends(PadreAppTest):

    def test_project(self):
        project = self.app.projects.create(name='Test Project deco',
                                           description='Testing the project decorators here.',
                                           creator=Function(fn=self.test_project))

        @project.decorator.experiment(name='Test Experiment', description='Test Experiment',
                                dataset=dataset.pop(), project=project,
                                pipeline=SKLearnPipeline(pipeline_fn=create_test_pipeline),
                                creator=self.test_all_functionalities)
        def custom_project():
            return


if __name__ == '__main__':
    unittest.main()
