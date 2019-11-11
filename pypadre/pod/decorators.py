import inspect
from functools import wraps

from pypadre.binding.model.sklearn_binding import SKLearnPipeline
from pypadre.core.model.generic.custom_code import _convert_path_to_code_object


# _experiments = {}


def workflow(*args, app, dataset, project_name=None, experiment_name=None, project_description=None, experiment_description=None, **kwargs):
    """
    Decroator for functions that return a single workflow to be executed in an experiment with name exp_name
    :param exp_name: name of the experiment
    :param args: additional positional parameters to an experiment (replaces other positional parameters if longer)
    :param kwargs: kwarguments for experiments
    :return:
    """
    def workflow_decorator(f_create_workflow):
        @wraps(f_create_workflow)
        def wrap_workflow(*args, **kwargs):
            # here the workflow gets called. We could add some logging etc. capability here, but i am not sure
            return f_create_workflow(*args, **kwargs)

        (filename, _, function_name, _, _) = inspect.getframeinfo(inspect.currentframe().f_back)
        creator = _convert_path_to_code_object(filename, function_name)

        pipeline = SKLearnPipeline(pipeline_fn=wrap_workflow, creator=creator)
        project = app.projects.create(name=project_name, description=project_description, creator=creator)
        experiment = app.experiments.create(name=experiment_name, description=experiment_description, project=project,
                                pipeline=pipeline, dataset=dataset, creator=creator)
        # TODO support parameters
        experiment.execute()
        return experiment

    return workflow_decorator


def dataset(*args, app, **kwargs):
    def dataset_decorator(f_create_dataset):
        @wraps(f_create_dataset)
        def wrap_dataset(*args, **kwargs):
            # here the workflow gets called. We could add some logging etc. capability here, but i am not sure
            return f_create_dataset(*args, **kwargs)

        return app.dataset.load(f_create_dataset())

    return dataset_decorator


# def component(*args, **kwargs):
#     def component_decorator(f_component):
#         @wraps(f_component)
#         def wrap_componet(*args, **kwargs):
#             # Register component
#             return f_component(*args, **kwargs)
#     return component_decorator
#
#
# def parameter_provider(*args, **kwargs):
#     def parameter_provider_decorator(f_pprovider):
#         @wraps(f_pprovider)
#         def wrap_pprovider(*args, **kwargs):
#             return f_pprovider(*args, **kwargs)
#     return parameter_provider_decorator
#
#
# def metric_provider(*args, **kwargs):
#     def metric_provider_decorator(f_mprovider):
#         @wraps(f_mprovider)
#         def wrap_mprovider(*args, **kwargs):
#             # Register component
#             return f_mprovider(*args, **kwargs)
#     return metric_provider_decorator
#
#
# def experiment(*args, **kwargs):
#     def experiment_decorator(f_experiment):
#         @wraps(f_experiment)
#         def wrap_experiment(*args, **kwargs):
#             return Experiment(*args, creator=f_experiment, **kwargs)
#         return wrap_experiment()
#     return experiment_decorator
#
#
# def project(*, name, description, **kwargs):
#     def project_decorator(f_project):
#         @wraps(f_project)
#         def wrap_project(*args, **kwargs):
#             p = Project(name=name, description=description, **kwargs)
#             p = f_project(*args, **kwargs)
#             p.send_put()
#             return p
#     return project_decorator
#
#
# def dataset(*args, **kwargs):
#     def dataset_decorator(f_dataset):
#         @wraps(f_dataset)
#         def wrap_dataset(*args, **kwargs):
#             # Register component
#             return f_dataset(*args, **kwargs)
#     return dataset_decorator
#
# # def merge_args(exp_name, args, kwargs):
# #     if len(_experiments[exp_name]["args"]) < len(args):
# #         _experiments[exp_name]["args"] = args  # taking the longer positional argument
# #     _experiments[exp_name]["kwargs"] = {**_experiments[exp_name]["kwargs"], **kwargs}
# #
# # # this is a decorator with an argument, that in principal replaces the constructor of padre Experiment
# # def Workflow(exp_name, *args, **kwargs):
# #     """
# #     Decroator for functions that return a single workflow to be executed in an experiment with name exp_name
# #     :param exp_name: name of the experiment
# #     :param args: additional positional parameters to an experiment (replaces other positional parameters if longer)
# #     :param kwargs: kwarguments for experiments
# #     :return:
# #     """
# #     def workflow_decorator(f_create_workflow):
# #         @wraps(f_create_workflow)
# #         def wrap_workflow(*args, **kwargs):
# #             # here the workflow gets called. We could add some logging etc. capability here, but i am not sure
# #             return f_create_workflow(*args, **kwargs)
# #         # here we store the workflow for the experiment in a dictionary usable as constructor to experiments
# #         # print("Storing workflow for experiment %s" % exp_name)
# #         if exp_name not in _experiments:
# #             _experiments[exp_name] = {"args": args, "kwargs": kwargs}
# #         else:
# #             merge_args(exp_name, args, kwargs)
# #         _experiments[exp_name]["workflow"] = wrap_workflow
# #         # now return the wrapped workflow.
# #         return wrap_workflow
# #
# #     return workflow_decorator
# #
# # # this is a decorator with an argument, that in principal replaces the constructor of padre Dataset
# # def Dataset(exp_name, *args, **kwargs):
# #     """
# #     Decorates a funtion that retuns a list of datasets
# #     :param exp_name: name of the experiment the datasets should be used with
# #     :param args: additional positional arguments for the experiment constructor
# #     :param kwargs: additional kw arguments for the expeirment constructor
# #     :return:
# #     """
# #     def dataset_decorator(f_get_datasets):
# #         @wraps(f_get_datasets)
# #         def wrap_dataset(*args, **kwargs):
# #             # here the workflow gets called. We could add some logging etc. capability here, but i am not sure
# #             print ("creating the workflow")
# #             return f_get_datasets(*args, **kwargs)
# #         # here we store the workflow for the experiment in a dictionary usable as constructor to experiments
# #         #print("Storing workflow for experiment %s" % exp_name)
# #         if exp_name not in _experiments:
# #             _experiments[exp_name] = {"args": args, "kwargs": kwargs}
# #         else:
# #             merge_args(exp_name, args, kwargs)
# #
# #         _experiments[exp_name]["datasets"] = wrap_dataset
# #         # now return the wrapped workflow.
# #         return wrap_dataset
# #
# #     return dataset_decorator
# #
# # def run(name=None, backend = None, change_name=True):
# #     """
# #     runs the experiments with the specific name. If no name is provided, all experiments available are run
# #     :param name: name of the experiment to run or None if all should be run
# #     :param backend: if not None, the backend will be set for all experiment runs
# #     :param change_name: if set to True, the experiment name is changed when hyperparameters are given.
# #     :return: Experiment object or list of Experiments if name was None.
# #     """
# #     def generate_config(hp_dict):
# #         """
# #         generator function that creates all possible configuration taking a key of a dict as parameter name and one
# #         value
# #         :param hp_dict:
# #         :return:
# #         """
# #         idx = []
# #         val = []
# #         # todo: dicts of dicts should be considered!!
# #         for k in hp_dict.keys():
# #             idx.append(k)
# #             val.append(hp_dict[k])
# #         for v in itertools.product(*val):
# #             yield dict(zip(idx, v))
# #
# #
# #
# #     def _run(name_, params_):
# #         if backend is not None:
# #             params_ = params_.copy()
# #             params_["kwargs"]["backend"] = backend
# #
# #         if "hyperparameters" in params_["kwargs"]:  # do we have to iterate over different hyperparameters?
# #             params_ = copy.deepcopy(params_)
# #             hp = params_["kwargs"].pop("hyperparameters")
# #             exs = []
# #             i = 0
# #             for config in generate_config(hp):
# #                 # if change_name:
# #                 #     # todo: do some template based labeling
# #                 #     # todo: we might introduce experiment groups (as sub directory or via labeling) in order to avoid the need for changing names.
# #                 #     # todo: we could also argue, that experiments are unique only within a single call fo _run or extend via UUID
# #                 #
# #                 #     n = name_+"("+",".join([str(k)+":"+str(v) for k, v in config.items()])+")"
# #                 # else:
# #                 #     n =name_
# #                 ex = Experiment(name=name_, **params_["kwargs"],
# #                                 workflow=params_["workflow"](**config),
# #                                 dataset=params_["datasets"]())
# #                 ex.run(append_runs=i > 0)
# #                 exs.append(ex)
# #                 i +=1
# #             return exs
# #
# #         else:
# #             ex = Experiment(name=name_, **params_["kwargs"],
# #                             workflow=params_["workflow"](),
# #                             dataset=params_["datasets"]())
# #             ex.run()
# #             return [ex]
# #
# #     if name is None:
# #         return [r for name_, params in _experiments.items() for r in _run(name_, params)]
# #
# #     else:
# #         if name not in _experiments:
# #             raise Exception("No experiment with name %s found. My config is: \n %s"
# #                             % (name, pprint.pformat(_experiments)))
# #         else:
# #             return [r for r in _run(name, _experiments[name])]
#
