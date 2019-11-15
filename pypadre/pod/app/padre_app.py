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

from docutils.nodes import warning
from jsonschema import ValidationError
from sklearn.pipeline import Pipeline

from pypadre._package import PACKAGE_ID
from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.model.code.code_mixin import PythonPackage, PythonFile, GitIdentifier, Function
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.core.model.pipeline.parameter_providers.parameters import ParameterProvider
from pypadre.core.util.utils import filter_nones, find_package_structure
from pypadre.pod.app.config.padre_config import PadreConfig
from pypadre.pod.app.core_app import CoreApp
from pypadre.pod.backend.file import PadreFileBackend
from pypadre.pod.backend.gitlab import PadreGitLabBackend


# logger = PadreLogger(app=None)
# add_logger(logger=logger)


class PadreAppFactory:

    @staticmethod
    def get(config=PadreConfig()):
        backends = PadreAppFactory._parse_backends(config)
        return PadreApp(backends=backends)

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
                #TODO check for validity
                backends.append(PadreGitLabBackend(b))
            elif 'root_dir' in b:
                # TODO check for validity
                backends.append(PadreFileBackend(b))
            else:
                raise ValidationError('{0} defined an invalid backend. Please provide either a http backend'
                                      ' or a local backend. (root_dir or base_url)'.format(b))
        return backends


class PadreApp(CoreApp):

    # ------------------------------------------ decorators -------------------------------------------
    def experiment(self, *args, ptype=None, parameters=None, parameter_provider=None,
                   reference=None, reference_package=None, reference_git=None,
                   dataset: Union[Dataset, str], project_name=None, experiment_name=None,
                   project_description=None, seed=None,
                   experiment_description=None, auto_main=True, **kwargs):
        """
        Decroator for functions that return a single workflow to be executed in an experiment with name exp_name
        :param args: additional positional parameters to an experiment (replaces other positional parameters if longer)
        :param ptype:
        :param parameters:
        :param parameter_provider: Object that provides parameters for hyperparameter search
        :param dataset: Dataset object of the experiment
        :param project_name:
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
                    pass
                # TODO look up the class by parsing the mapping / looking at the return value of the function or something similar
                # raise NotImplementedError()

            else:
                local_ptype = ptype

            # TODO check pipeline type (where to put provider)
            if parameter_provider is not None:
                pipeline = local_ptype(pipeline_fn=wrap_workflow, parameter_provider=parameter_provider,
                                       reference=creator)
            else:
                pipeline = local_ptype(pipeline_fn=wrap_workflow, reference=creator)

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
