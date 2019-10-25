# from pypadre.core.sklearnworkflow import SKLearnWorkflow

from pypadre.core.base import MetadataEntity, ChildEntity
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.core.model.execution import Execution
from pypadre.core.model.generic.custom_code import ICodeManagedObject
from pypadre.core.model.generic.i_executable_mixin import IExecuteable
from pypadre.core.model.generic.i_model_mixins import IStoreable, IProgressable
from pypadre.core.model.pipeline.pipeline import Pipeline
from pypadre.core.model.project import Project


####################################################################################################################
#  Module Private Functions and Classes
####################################################################################################################


def _sklearn_runner():
    pass


def _is_sklearn_pipeline(pipeline):
    """
    checks whether pipeline is a sklearn pipeline
    :param pipeline:
    :return:
    """
    # we do checks via strings, not isinstance in order to avoid a dependency on sklearn
    return type(pipeline).__name__ == 'Pipeline' and type(pipeline).__module__ == 'sklearn.pipeline'


class Experiment(ICodeManagedObject, IStoreable, IProgressable, IExecuteable, MetadataEntity, ChildEntity):
    """
    Experiment class covering functionality for executing and evaluating machine learning experiments.
    It is determined by a pipeline which is evaluated over a dataset with several configuration.
    A run applies one configuration over the data, which can be splitted in several sub-runs on different dataset parts
    in order to get reliable statistical estimates.

    An experiment requires:
    1. a pipeline / workflow. A workflow implements `fit`, `infer` and `transform` methods, comparable to sklearn.
    Currently, we only support sklearn pipelines, which are wrapped by the SKLearnWorkflow
    <http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>,
    i.e. a list of (name, class) tuples, where the class implements:
       - a `fit` function (parameters need to be defined)
       - a `infer` function in case of supervised prediction (parameters need to be defined)
       - a `transform`function in case of feature space transformers (parameters need to be defined)

    2. a dataset. An experiment is always tight to one dataset which is the main dataset for running the experiment.
       Future work should allow to include auxiliary resources, but currently we only support one dataset.
    3. metadata describing different aspects of the workflow.
      - the splitting strategy (see Splitter)
      - hyperparameter ranges
      - output control etc.


    Experiment Metadata:
    ====================

    All metadata provided to the experiment will be stored along the experiment description. However, the following
    properties will gain special purpose for an experiment:
    - task - determines the task achieved by a experiment (e.g. classification, regression, metric learning etc.)
    - name - determines the name of an experiment
    - id - determines the repository id of an experiment (might be equal to the name, if the name is also the id)
    - description - determines the description of an experiment
    - domain - determines the application domain

    Parameters required:
    ===================
    The following parameters need to be set in the constructor or via annotations
    - dataset : pypadre.datasets.Dataset
    - workflow: either a pypadre.experiment.Workflow object or a SKLearn Pipeline

    Options supported:
    ==================
    - stdout={True|False} logs event messages to default_logger. Default = True
    - keep_splits={True|False} if true, all split data for every run is kept (including the model,
                                   split inidices and training data) are kept in memory.
                               If false, no split data is kept
    - keep_runs={True|False} if true, all rund data (i.e. scores) will be kept in memory.
                             If false, no split run data is not kept
    - n_runs = int  number of runs to conduct. todo: needs to be extended with hyperparameter search

    TODO:
    - Queuing mode
    - Searching Hyperparameter Space

    """

    PROJECT_ID = "project_id"
    DATASET_ID = "dataset_id"
    NAME = "name"
    DESCRIPTION = "description"
    # variable to store the path of the source code which is used to run the experiment
    CODE_PATH = 'code_path'

    # TODO non-metadata input should be a parameter
    def __init__(self, name, description, project: Project = None, dataset: Dataset = None, pipeline: Pipeline = None, **kwargs):
        # Add defaults
        defaults = {"name": "default experiment name", "description": "This is the default experiment."}

        # Either get given path or look up the path of the calling file

        # Merge defaults
        metadata = {**defaults, **kwargs.pop("metadata", {}), **{
            self.PROJECT_ID: project.id if project else None,
            self.DATASET_ID: dataset.id if dataset else None,
            self.NAME: name,
            self.DESCRIPTION: description
        }}

        super().__init__(parent=project, schema_resource_name="experiment.json",
                         metadata=metadata, **kwargs)
        # Variables
        self._dataset = dataset
        self._pipeline = pipeline
        self._executions = []

    @property
    def project(self):
        return self.parent

    @property
    def dataset(self):
        return self._dataset

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def executions(self):
        return self._executions

    def _execute_helper(self, *args, **kwargs):

        if self.pipeline is None:
            raise ValueError("Pipeline has to be defined to run an experiment")
        if self.dataset is None:
            raise ValueError("Dataset has to be defined to run an experiment")
        # TODO command
        # TODO check pipeline_parameters mapping to components?

        # dict_object = {'path': self.metadata.get(self.CODE_PATH),
        #                'create_repo': False}

        # Get hash from the outside if possible
        code_hash = self.creator_hash

        # Should we simply warn the user that there is no repository for the code
        if code_hash is None:
            raise ValueError("An execution has to reference a code hash to be valid.")

        execution = Execution(experiment=self, codehash=code_hash, command=kwargs.pop("cmd", "default"))
        self._executions.append(execution)
        return execution.execute(**kwargs)
