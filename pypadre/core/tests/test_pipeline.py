# import unittest
#
# # noinspection PyMethodMayBeStatic
# import numpy as np
#
# from pypadre.core.model.dataset.dataset import Dataset
# from pypadre.core.model.experiment import Experiment
# from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
# from pypadre.core.ontology.padre_ontology import PaDREOntology
#
# test_numpy_array = np.array([[1.0, "A", 2],
#                              [2.0, "B", 2],
#                              [3.0, "A", 3],
#                              [3.0, "C", 4]])
#
#
# class Pipeline(unittest.TestCase):
#
#     def __init__(self, *args, **kwargs):
#         super(Pipeline, self).__init__(*args, **kwargs)
#
#     def test_default_python_pipeline(self):
#         def splitting(*, data, **kwargs):
#             print(data)
#             return [[[1.0, "A", 2], [2.0, "B", 2]], [[3.0, "A", 3], [3.0, "C", 4]]]
#
#         def estimator(*, data, **kwargs):
#             data = "estimate:" + str(data)
#             print(data)
#             return data
#
#         def evaluator(*, data, **kwargs):
#             data = "evaluate:" + str(data)
#             print(data)
#             return data
#
#         pipeline = DefaultPythonExperimentPipeline(splitting=splitting, estimator=estimator,
#                                                    evaluator=evaluator)
#
#         dataset = Dataset(name="A name", description="A description",
#                           type=PaDREOntology.SubClassesDataset.Multivariat.value)
#         dataset.set_data(test_numpy_array)
#
#         experiment = Experiment(dataset=dataset, pipeline=pipeline)
#
#         experiment.execute()
#         # TODO asserts and stuff
#
#
# if __name__ == '__main__':
#     unittest.main()
