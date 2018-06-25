"""
Command Line Interface for PADRE.

"""
# todo support config file https://stackoverflow.com/questions/46358797/python-click-supply-arguments-and-options-from-a-configuration-file
import click
import padre.app.padre_app as app
from padre.app import pypadre


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
    padre command line interface.

    Default config file: ~/.padre.cfg
    """
    # load default config
    config = app.default_config.copy()
    # override defaults
    if base_url is not None:
        config["HTTP"]["base_url"] = base_url
    # create app objects
    _http = app.http_client
    _file = app.file_cache
    # create context object
    ctx.obj = {
        'config-file': config_file,
        'API': config["HTTP"],
        'http-client': _http,
        'pypadre': app.PadreApp(_http, _file, click.echo)
    }


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
#@click.option(default="linear regression", help='Shows all parameters of an estimator')
@click.pass_context
def components(ctx, estimator):
    print(ctx.obj["pypadre"].experiment_creator.get_estimator_params(estimator))


@pypadre_cli.command(name="create_experiment")
@click.option('--name', default=None, help='Name of the experiment. If none UUID will be given')
@click.option('--description', default=None, help='Description of the experiment')
@click.option('--dataset', default=None, help='Name of the dataset to be used in the experiment')
@click.option('--workflow', default=None, help='Estimators to be used in the workflow')
@click.option('--backend', default=None, help='Backend of the experiment')
@click.pass_context
def create_experiment(ctx, name, description, dataset, workflow, backend):
    estimator_list = (workflow.replace(", ",",")).split(sep=",")
    workflow_obj = ctx.obj["pypadre"].experiment_creator.create_test_pipeline(estimator_list)
    if backend is None:
        backend = pypadre.file_repository.experiments
    ctx.obj["pypadre"].experiment_creator.create_experiment(name, description, dataset, workflow_obj, backend)


@pypadre_cli.command(name="run")
@click.pass_context
def execute(ctx):
    ctx.obj["pypadre"].experiment_creator.execute_experiments()


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


if __name__ == '__main__':
    pypadre_cli()
