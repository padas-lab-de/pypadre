import pprint
from functools import wraps
from padre.experiment import Experiment

_experiments = {}


def merge_args(exp_name, args, kwargs):
    if len(_experiments[exp_name]["args"]) < len(args):
        _experiments[exp_name]["args"] = args  # taking the longer positional argument
    _experiments[exp_name]["kwargs"] = {**_experiments[exp_name]["kwargs"], **kwargs}


# this is a decorator with an argument, that in principal replaces the constructor of padre Experiment
def Workflow(exp_name, *args, **kwargs):
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
        # here we store the workflow for the experiment in a dictionary usable as constructor to experiments
        print("Storing workflow for experiment %s" % exp_name)
        if exp_name not in _experiments:
            _experiments[exp_name] = {"args": args, "kwargs": kwargs}
        else:
            merge_args(exp_name, args, kwargs)
        _experiments[exp_name]["workflow"] = wrap_workflow
        # now return the wrapped workflow.
        return wrap_workflow

    return workflow_decorator

# this is a decorator with an argument, that in principal replaces the constructor of padre Experiment
def Dataset(exp_name, *args, **kwargs):
    """
    Decorates a funtion that retuns a list of datasets
    :param exp_name: name of the experiment the datasets should be used with
    :param args: additional positional arguments for the experiment constructor
    :param kwargs: additional kw arguments for the expeirment constructor
    :return:
    """
    def dataset_decorator(f_get_datasets):
        @wraps(f_get_datasets)
        def wrap_dataset(*args, **kwargs):
            # here the workflow gets called. We could add some logging etc. capability here, but i am not sure
            print ("creating the workflow")
            return f_get_datasets(*args, **kwargs)
        # here we store the workflow for the experiment in a dictionary usable as constructor to experiments
        print("Storing workflow for experiment %s" % exp_name)
        if exp_name not in _experiments:
            _experiments[exp_name] = {"args": args, "kwargs": kwargs}
        else:
            merge_args(exp_name, args, kwargs)

        _experiments[exp_name]["datasets"] = wrap_dataset
        # now return the wrapped workflow.
        return wrap_dataset

    return dataset_decorator

def run(name=None, backend = None):
    """
    runs the experiments with the specific name. If no name is provided, all experiments available are run
    :param name: name of the experiment to run or None if all should be run
    :return: Experiment object or list of Experiments if name was None.
    """
    def _run(name_, params_):
        if backend is not None:
            params_ = params_.copy()
            params_["backend"] = backend
        ex = Experiment(name=name_, **params_["kwargs"],
                        workflow=params_["workflow"](),
                        dataset=params_["datasets"]())
        ex.run()
        return ex

    if name is None:
        return [_run(name_, params) for name_, params in _experiments.items()]

    else:
        if name not in _experiments:
            raise Exception("No experiment with name %s found. My config is: \n %s"
                            % (name, pprint.pformat(_experiments)))
        else:
            return _run(name,_experiments[name])

