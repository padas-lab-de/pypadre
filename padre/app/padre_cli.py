"""
Command Line Interface for PADRE.

"""
# todo support config file https://stackoverflow.com/questions/46358797/python-click-supply-arguments-and-options-from-a-configuration-file
import click
from padre.app import pypadre, PadreApp, PadreConfig

#################################
#######      MAIN      ##########
#################################
@click.group()
@click.option('--config-file', '-c',
              type=click.Path(),
              default='~/.padre.cfg',
              )
@click.option('--base-url', '-b', envvar="PADRE_URL",
              type=str,
              help="Base url for the PADRE REST API")
@click.pass_context
def pypadre_cli(ctx, config_file, base_url):
    """
    setup padre command line interface using the provided config file

    Default config file: ~/.padre.cfg
    """
    # load default config
    config = PadreConfig(config_file)
    # override defaults
    if base_url is not None:
        config["HTTP BACKEND"]["base_url"] = base_url
    # create context object
    ctx.obj = {
        'config-file': config_file,
        'pypadre': PadreApp(config, click.echo)
    }


@pypadre_cli.command(name="get_config_param")
@click.option('--param', default=None, help='Get value of given param')
@click.pass_context
def get_config_param(ctx, param):
    """
    Get given param from config file.
    """
    result = ctx.obj["pypadre"].config.get(param)
    print(result)


@pypadre_cli.command(name="set_config_param")
@click.option('--param', nargs=2, default=None, help='key value pair as tuple')
@click.pass_context
def set_config_param(ctx, param):
    """
    Sets key, value in config. param must be a tuple
    """
    ctx.obj["pypadre"].config.set(param[0], param[1])


@pypadre_cli.command(name="list_config_params")
@click.pass_context
def list_config_params(ctx):
    """
    List all values in config
    """
    result = ctx.obj["pypadre"].config.list()
    print(result)


@pypadre_cli.command(name="authenticate")
@click.option('--url', default=None, help='Url of server api')
@click.option('--user', default=None, help='User on server')
@click.option('--passwd', default=None, help='Password for given user')
@click.pass_context
def authenticate(ctx, url, user, passwd):
    """
    To generate new token in config. Authenticate with given credentials, in case credentials
    are not provided default credentials will be used.
    """
    ctx.obj["pypadre"].config.authenticate(url, user, passwd)


#################################
####### DATASETS FUNCTIONS ##########
#################################


@pypadre_cli.command(name="datasets")
@click.option('--start', default=0, help='start number of the dataset')
@click.option('--count', default=999999999, help='Number of datasets to retrieve')
@click.option('--search', default=None,
              help='search string')
@click.pass_context
def datasets(ctx, start, count, search):
    """list all available datasets"""
    ctx.obj["pypadre"].datasets.list_datasets(start, count, search)


@pypadre_cli.command(name="import")
@click.option('--sklearn/--no-sklearn', default=True, help='import sklearn default datasets')
@click.pass_context
def do_import(ctx, sklearn):
    """import default datasets from different sources specified by the flags"""
    ctx.obj["pypadre"].datasets.do_default_imports(sklearn)


@pypadre_cli.command(name="dataset")
@click.argument('dataset_id')
@click.option('--binary/--no-binary', default=True, help='download binary')
@click.option('--format', '-f', default="numpy", help='format for binary download')
@click.pass_context
def dataset(ctx, dataset_id, binary, format):
    """downloads the dataset with the given id. id can be either a number or a valid url"""
    ctx.obj["pypadre"].datasets.get_dataset(dataset_id, binary, format)

@pypadre_cli.command(name="oml_dataset")
@click.argument('dataset_id')
@click.pass_context
def dataset(ctx, dataset_id):
    """downloads the dataset with the given id. id can be either a number or a valid url"""
    ctx.obj["pypadre"].datasets.get_openml_dataset(dataset_id)



@pypadre_cli.command(name="upload_scratchdata_multi")
@click.pass_context
def dataset(ctx):
    """uploads sklearn toy-datasets, top 100 Datasets form OpenML and some Graphs
    Takes about 1 hour
    Requires about 7GB of Ram"""

    auth_token=ctx.obj["pypadre"].config.get("token")

    ctx.obj["pypadre"].datasets.upload_scratchdatasets(auth_token,max_threads=8,upload_graphs=True)


@pypadre_cli.command(name="upload_scratchdata_single")
@click.pass_context
def dataset(ctx):
    """uploads sklearn toy-datasets, top 100 Datasets form OpenML and some Graphs
    Takes several hours
    Requires up to 7GB of Ram"""

    auth_token = ctx.obj["pypadre"].config.get("token")

    ctx.obj["pypadre"].datasets.upload_scratchdatasets(auth_token, max_threads=1, upload_graphs=True)


#################################
####### EXPERIMENT FUNCTIONS ##########
#################################

@pypadre_cli.command(name="experiment")
@click.pass_context
def show_experiments(ctx):
    # List all the experiments that are currently saved
    ctx.obj["pypadre"].experiment_creator.experiment_names


@pypadre_cli.command(name="components")
@click.pass_context
def show_components(ctx):
    # List all the components of the workflow
    print(ctx.obj["pypadre"].experiment_creator.components)


@pypadre_cli.command(name="parameters")
@click.argument('estimator')
@click.pass_context
def components(ctx, estimator):
    print(ctx.obj["pypadre"].experiment_creator.get_estimator_params(estimator))


@pypadre_cli.command(name="datasets")
@click.pass_context
def datasets(ctx):
    print(ctx.obj["pypadre"].experiment_creator.get_dataset_names())


@pypadre_cli.command(name='set_params')
@click.option("--experiment", default=None, help='Name of the experiment to which parameters are to be set.')
@click.option("--parameters", default=None, help='Name of the parameter and the parameters.')
@click.pass_context
def set_parameters(ctx, experiment, parameters):
    ctx.obj["pypadre"].experiment_creator.set_param_values(experiment, parameters)


@pypadre_cli.command(name='get_params')
@click.option("--experiment", default=None, help='Name of the experiment from which parameters are to be retrieved')
@click.pass_context
def get_parameters(ctx, experiment):
    print(ctx.obj["pypadre"].experiment_creator.get_param_values(experiment))


@pypadre_cli.command(name="create_experiment")
@click.option('--name', default=None, help='Name of the experiment. If none UUID will be given')
@click.option('--description', default=None, help='Description of the experiment')
@click.option('--dataset', default=None, help='Name of the dataset to be used in the experiment')
@click.option('--workflow', default=None, help='Estimators to be used in the workflow')
@click.option('--backend', default=None, help='Backend of the experiment')
@click.pass_context
def create_experiment(ctx, name, description, dataset, workflow, backend):
    workflow_obj = None
    if workflow is not None:
        estimator_list = (workflow.replace(", ",",")).split(sep=",")
        workflow_obj = ctx.obj["pypadre"].experiment_creator.create_test_pipeline(estimator_list)

    if backend is None:
        backend = pypadre.local_backend.experiments
    ctx.obj["pypadre"].experiment_creator.create_experiment(name, description, dataset, workflow_obj, backend)


@pypadre_cli.command(name="run")
@click.pass_context
def execute(ctx):
    ctx.obj["pypadre"].experiment_creator.execute_experiments()


@pypadre_cli.command(name="do_experiments")
@click.option('--experiments', default=None, help='Names of the experiments where the datasets should be applied')
@click.option('--datasets', default=None, help="Names of datasets for each experiment separated by ;")
@click.pass_context
def do_experiment(ctx, experiments, datasets):
    import copy
    datasets_list = datasets.split(sep=";")
    experiments_list = experiments.split(sep=",")
    if len(datasets_list) == len(experiments_list):
        experiment_datasets_dict = dict()
        for idx in range(0, len(experiments_list)):
            datasets_list[idx] = ((datasets_list[idx].strip()).replace(", ", ",")).replace(" ,",",")
            curr_exp_datasets = datasets_list[idx].split(sep=",")
            experiment_datasets_dict[experiments_list[idx]] = copy.deepcopy(curr_exp_datasets)

        ctx.obj["pypadre"].experiment_creator.do_experiments(experiment_datasets_dict)


@pypadre_cli.command(name="load_config_file")
@click.option('--filename', default=None, help='Path of the JSON file that contains the experiment parameters')
@click.pass_context
def load_config_file(ctx, filename):

    import os

    if os.path.exists(filename):
        ctx.obj["pypadre"].experiment_creator.parse_config_file(filename)
        ctx.obj["pypadre"].experiment_creator.execute_experiments()

    else:
        print('File does not exist')



#################################
####### METRICS FUNCTIONS ##########
#################################

@pypadre_cli.command(name="compare_metrics")
@click.option('--path', default=None, help='Path of the experiment whose runs are to be compared')
@click.option('--query', default="all", help="Results to be displayed based on the runs")
@click.option('--metrics', default=None, help='Metrics to be displayed')
@click.pass_context
def compare_runs(ctx, path, query, metrics):
    metrics_list = None
    dir_path_list = (path.replace(", ","")).split(sep=",")
    estimators_list = (query.replace(", ","")).split(sep=",")
    if metrics is not None:
        metrics_list = (metrics.replace(", ","")).split(sep=",")
    ctx.obj["pypadre"].metrics_evaluator.read_run_directories(dir_path_list)
    ctx.obj["pypadre"].metrics_evaluator.get_unique_estimators_parameter_names()
    ctx.obj["pypadre"].metrics_evaluator.read_split_metrics()
    ctx.obj["pypadre"].metrics_evaluator.analyze_runs(estimators_list, metrics_list)
    print(ctx.obj["pypadre"].metrics_evaluator.display_results())


@pypadre_cli.command(name="reevaluate_metrics")
@click.option('--path', default=None, help='Path of experiments whose metrics are to be reevaluated')
@click.pass_context
def reevaluate_runs(ctx, path):
    path_list = (path.replace(", ", ",")).split(sep=",")
    ctx.obj["pypadre"].metrics_reevaluator.get_split_directories(dir_path=path_list)
    ctx.obj["pypadre"].metrics_reevaluator.recompute_metrics()


@pypadre_cli.command(name="list_experiments")
@click.pass_context
def list_experiments(ctx):
    print(ctx.obj["pypadre"].metrics_evaluator.get_experiment_directores())


@pypadre_cli.command(name='get_available_estimators')
@click.pass_context
def get_available_estimators(ctx):
    estimators = ctx.obj["pypadre"].metrics_evaluator.get_unique_estimator_names()
    if estimators is None:
        print('No estimators found for the runs compared')
    else:
        print(estimators)


@pypadre_cli.command(name="list_estimator_params")
@click.option('--estimator_name', default=None, help='Params of this estimator will be displayed')
@click.option('--selected_params', default=None, help='List the values of only these parameters')
@click.option('--return_all_values', default=None,
              help='Returns all the parameter values for the estimator including the default values')
def list_estimator_params(ctx, estimator_name, selected_params, return_all_values):
    params_list = \
        ctx.obj["pypadre"].metrics_evaluator.get_estimator_param_values\
            (estimator_name, selected_params, return_all_values)
    if params_list is None:
        print('No parameters obtained')
    else:
        for param in params_list:
            print(param, params_list.get(param, '-'))


if __name__ == '__main__':
    pypadre_cli()
