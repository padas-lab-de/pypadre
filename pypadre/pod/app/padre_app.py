# TODO DELETE AS SOON AS TRANSFERED INTO NEW APP CLASSES

"""
Padre app as single point of interaction.

Defaults:

- The default configuration is provided under `.padre.cfg` in the user home directory


Architecture of the module
++++++++++++++++++++++++++

- `PadreConfig` wraps the configuration for the app. It can read/write the config from a file (if provided)

"""

# todo merge with cli. cli should use app and app should be configurable via builder pattern and configuration files
import inspect
from functools import wraps
from logging import warning
from pathlib import Path
from typing import Union

import numpy as np
from docutils.nodes import warning
from jsonschema import ValidationError
from sklearn.pipeline import Pipeline

from pypadre._package import PACKAGE_ID
from pypadre.binding.model.sklearn_binding import SKLearnPipeline, SKLearnEvaluator
from pypadre.core.model.code.code_mixin import PythonPackage, PythonFile, GitIdentifier, Function
from pypadre.core.model.computation.evaluation import Evaluation
from pypadre.core.model.computation.training import Training
from pypadre.core.model.dataset.dataset import Dataset, Transformation
from pypadre.core.model.pipeline.parameter_providers.parameters import ParameterProvider
from pypadre.core.model.pipeline.pipeline import DefaultPythonExperimentPipeline
from pypadre.core.model.split.split import Split
from pypadre.core.util.utils import filter_nones, find_package_structure, unpack
from pypadre.pod.app.config.padre_config import PadreConfig
from pypadre.pod.app.core_app import CoreApp
from pypadre.pod.backend.file import PadreFileBackend
from pypadre.pod.backend.gitlab import PadreGitLabBackend


# logger = PadreLogger(app=None)
# add_logger(logger=logger)


class PadreAppFactory:

    @staticmethod
    def get(config=None, printer=print):
        if config is None:
            config = PadreConfig()
        backends = PadreAppFactory._parse_backends(config)
        return PadreApp(config=config, printer=printer, backends=backends)

    @staticmethod
    def _parse_backends(config):
        _backends = config.get("backends", "GENERAL")

        backends = []
        for b in _backends:
            if 'base_url' in b:
                # TODO check for validity
                pass
                # backends.append(PadreHttpBackend(b))
            elif 'gitlab_url' in b:
                # TODO check for validity
                backends.append(PadreGitLabBackend(b))
            elif 'root_dir' in b:
                # TODO check for validity
                backends.append(PadreFileBackend(b))
            else:
                raise ValidationError('{0} defined an invalid backend. Please provide either a http backend'
                                      ' or a local backend. (root_dir or base_url)'.format(b))
        return backends


class PadreApp(CoreApp):

    def __init__(self, config, **kwargs):
        self._config = config
        super().__init__(**kwargs)

    @property
    def config(self):
        return self._config

    # ------------------------------------------ decorators -------------------------------------------
    def experiment(self, *args, ptype=None, parameters=None, parameter_provider=None, splitting=None,
                   preprocessing_fn=None, reference=None, reference_package=None, reference_git=None,
                   dataset: Union[Dataset, str], project_name=None, experiment_name=None,
                   project_description=None, seed=None, estimator=None, evaluator=None,
                   experiment_description=None, auto_main=True, **kwargs):
        """
        Decorator for functions that return a single workflow to be executed in an experiment with name exp_name
        :param args: additional positional parameters to an experiment (replaces other positional parameters if longer)
        :param ptype: Pipeline type
        :param parameters: parameters of the passed pipeline
        :param parameter_provider: Object that provides parameters for hyperparameter search
        :param dataset: Dataset object of the experiment
        :param project_name: Name of the project
        :param experiment_name: Name of the experiment
        :param project_description:
        :param experiment_description:
        :param auto_main:
        :param kwargs: kwarguments for experiments
        :return:
        """

        if parameters is None:
            parameters = {}

        def experiment_decorator(f_create_experiment):
            @wraps(f_create_experiment)
            def wrap_workflow(*args, **kwargs):
                # here the workflow gets called. We could add some logging etc. capability here, but i am not sure
                return f_create_experiment(*args, **kwargs)

            creator = to_decorator_reference(reference, reference_package, reference_git)

            local_ptype = None
            if ptype is None:
                pipeline = wrap_workflow()
                if isinstance(pipeline, Pipeline):
                    # TODO plugin don't reference the binding here!!!!
                    local_ptype = SKLearnPipeline
                else:
                    local_ptype = DefaultPythonExperimentPipeline
                # TODO look up the class by parsing the mapping / looking at the return value of the function or something similar
                # raise NotImplementedError()

            else:
                local_ptype = ptype

            # TODO check pipeline type (where to put provider)
            if local_ptype == SKLearnPipeline:
                pipeline = local_ptype(pipeline_fn=wrap_workflow, splitting=splitting,
                                       preprocessing_fn=preprocessing_fn,
                                       parameter_provider=parameter_provider,
                                       reference=creator)
            else:
                pipeline = local_ptype(splitting=splitting, estimator=estimator, evaluator=evaluator,
                                       preprocessing_fn=preprocessing_fn,
                                       reference=creator)

            project = self.projects.get_by_name(project_name)
            if project is None:
                project = self.projects.create(
                    **filter_nones({"name": project_name, "description": project_description}), reference=creator)

            d = dataset if isinstance(dataset, Dataset) else self.datasets.get_by_name(dataset)
            experiment = self.experiments.create(
                **filter_nones({"name": experiment_name, "description": experiment_description}),
                project=project, pipeline=pipeline, dataset=d, reference=creator, seed=seed)
            if auto_main:
                experiment.execute(parameters=parameters)
                return experiment
            else:
                if parameters:
                    warning("Parameters are given but experiment is not started directly. Parameters will be omitted. "
                            "You have to pass them on the execute call again.")
                return experiment

        return experiment_decorator

    def parameter_map(self):
        def parameter_decorator(f_create_parameters):
            @wraps(f_create_parameters)
            def wrap_parameters(*args, **kwargs):
                # here the parameter map gets called. We could add some logging etc. capability here,
                # but i am not sure
                return f_create_parameters(*args, **kwargs)

            return wrap_parameters()

        return parameter_decorator

    def parameter_provider(self, *args, reference=None, reference_package=None, **kwargs):
        def parameter_decorator(f_create_parameters):
            @wraps(f_create_parameters)
            def wrap_parameters(*args, **kwargs):
                # here the parameter provider gets called. We could add some logging etc. capability here,
                # but i am not sure
                return f_create_parameters(*args, **kwargs)

            creator = to_decorator_reference(reference, reference_package)

            return ParameterProvider(name="custom_parameter_provider", reference=creator,
                                     code=Function(fn=wrap_parameters, identifier=creator.identifier, transient=True))

        return parameter_decorator

    def dataset(self, *args, name=None, **kwargs):
        def dataset_decorator(f_create_dataset):
            @wraps(f_create_dataset)
            def wrap_dataset(*args, **kwargs):
                # here the workflow gets called. We could add some logging etc. capability here, but i am not sure
                return f_create_dataset(*args, **kwargs)

            if name is None:
                return self.datasets.load(wrap_dataset())
            return self.datasets.load(wrap_dataset(), name=name, **kwargs)

        return dataset_decorator

    def custom_splitter(self, *args, reference_git=None, reference_package=None, **kwargs):
        def splitter_decorator(f_create_splitter):
            @wraps(f_create_splitter)
            def wrap_splitter(*args, **kwargs):
                # here the custom splitter get called.
                num = -1
                (data, run, component, predecessor) = unpack(args[0], "data", "run", "component", ("predecessor", None))
                train_idx, test_idx, val_idx = f_create_splitter(data, **kwargs)
                yield Split(run=run, num=++num, train_idx=train_idx, test_idx=test_idx,
                            val_idx=val_idx, component=component, predecessor=predecessor, **kwargs)

            creator = to_decorator_reference(reference_git=reference_git, reference_package=reference_package)
            return Function(fn=wrap_splitter, transient=True, identifier=creator.identifier, **kwargs)

        return splitter_decorator

    def preprocessing(self, *args, reference_git=None, reference_package=None, store=False, **kwargs):
        def preprocessing_decorator(f_create_preprocessing):
            @wraps(f_create_preprocessing)
            def wrap_preprocessing(*args, **kwargs):
                (dataset,) = unpack(args[0], "data")
                _data = f_create_preprocessing(dataset, **kwargs)
                _dataset = Transformation(name="Standarized_%s" % dataset.name, dataset=dataset)
                _dataset.set_data(_data, attributes=dataset.attributes)
                if store:
                    self.datasets.put(_dataset)
                return _dataset

            creator = to_decorator_reference(reference_git=reference_git, reference_package=reference_package)

            return Function(fn=wrap_preprocessing, transient=True, identifier=creator.identifier, **kwargs)

        return preprocessing_decorator

    def estimator(self, *args, config=None, reference_git=None, reference_package=None, **kwargs):
        def estimator_decorator(f_create_estimator):
            @wraps(f_create_estimator)
            def wrap_estimator(*args, **kwargs):
                (split, component, run, initial_hyperparameters) = unpack(args[0], "data", "component", "run",
                                                                          "initial_hyperparameters")
                y = None
                if split.train_targets is not None:
                    y = split.train_targets.reshape((len(split.train_targets),))
                else:
                    y = np.zeros(shape=(len(split.train_features, )))
                model = f_create_estimator(split.train_features, y, **config)

                return Training(split=split, component=component, run=run, model=model, parameters=config,
                                initial_hyperparameters=initial_hyperparameters)

            creator = to_decorator_reference(reference_package=reference_package, reference_git=reference_git)
            return Function(fn=wrap_estimator, transient=True, identifier=creator.identifier)

        return estimator_decorator

    def evaluator(self, *args, task_type=None, reference_git=None, reference_package=None, **kwargs):
        def evaluator_decorator(f_create_evaluator):
            @wraps(f_create_evaluator)
            def wrap_evaluator(*args, **kwargs):
                data, predecessor, component, run = unpack(args[0], "data", ("predecessor", None), "component", "run")
                model = data["model"]
                split = data["split"]
                if not split.has_testset():
                    raise ValueError("Test set is missing")
                train_idx = split.train_idx.tolist()
                test_idx = split.test_idx.tolist()
                y = split.test_targets.reshape((len(split.test_targets),))
                X_test = split.test_features
                y_pred, probabilities = f_create_evaluator(model, X_test, **kwargs)
                results = SKLearnEvaluator.create_results_dictionary(split_num=split.number, train_idx=train_idx,
                                                                     test_idx=test_idx,
                                                                     dataset=split.dataset.name,
                                                                     truth=y.tolist(), predicted=y_pred.tolist(),
                                                                     type_=task_type,
                                                                     probabilities=probabilities.tolist())

                return Evaluation(training=predecessor, result_format=task_type, result=results, component=component,
                                  run=run,
                                  parameters=kwargs)

            creator = to_decorator_reference(reference_package=reference_package, reference_git=reference_git)
            return Function(fn=wrap_evaluator, transient=True, identifier=creator.identifier)

        return evaluator_decorator


def to_decorator_reference(reference=None, reference_package=None, reference_git=None):
    if reference is not None:
        creator = reference
    elif reference_package is not None:
        (filename, _, function_name, _, _) = inspect.getframeinfo(inspect.currentframe().f_back.f_back)
        creator = PythonPackage(package=find_package_structure(reference_package), variable=function_name,
                                identifier=PACKAGE_ID)
    elif reference_git is not None:
        (filename, _, function_name, _, _) = inspect.getframeinfo(inspect.currentframe().f_back.f_back)
        # TODO find git repo or pass it and look where package starts import from there
        git_repo = str(Path(reference_git).parent)
        creator = PythonFile(path=str(Path(reference_git).parent), package=reference_git[len(git_repo) + 1:],
                             variable=function_name,
                             identifier=GitIdentifier(path=git_repo))
    else:
        raise ValueError("You need to provide a reference for your definition.")
    return creator
