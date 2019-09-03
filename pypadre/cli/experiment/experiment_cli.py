"""
Command Line Interface for PADRE.

"""
import click


#################################
####### EXPERIMENT FUNCTIONS ##########
#################################

@click.group(name="experiment", invoke_without_command=True)
@click.option('--id', '-i', type=str, help="id of the new default experiment")
@click.pass_context
def experiment(ctx):
    """
    Commands for experiments.
    """
    if ctx.invoked_subcommand is None:
        if ctx.obj["pypadre-app"].config.get("experiment", "DEFAULTS") is not None:
            click.echo('Current default project is ' + ctx.obj["pypadre-app"].config.get("experiment", "DEFAULTS"))


@experiment.command(name="select")
@click.argument('experiment_name')
@click.pass_context
def select(ctx, id):
    # Set as active project
    ctx.obj["pypadre-app"].config.set("experiment", id, "DEFAULTS")


@experiment.command(name="list")
@click.pass_context
def list(ctx):
    # List all the experiments that are currently saved
    print(ctx.obj["pypadre-app"].experiment_creator.experiment_names)


@experiment.command(name="components")
@click.pass_context
def show_components(ctx):
    # List all the components of the workflow
    print(ctx.obj["pypadre-app"].experiment_creator.components)


@experiment.command(name="parameters")
@click.argument('estimator')
@click.pass_context
def components(ctx, estimator):
    print(ctx.obj["pypadre-app"].experiment_creator.get_estimator_params(estimator))


@experiment.command(name="datasets")
@click.pass_context
def datasets(ctx):
    print(ctx.obj["pypadre-app"].experiment_creator.get_dataset_names())


@experiment.command(name='set_params')
@click.option("--experiment", default=None, help='Name of the experiment to which parameters are to be set.')
@click.option("--parameters", default=None, help='Name of the parameter and the parameters.')
@click.pass_context
def set_parameters(ctx, experiment, parameters):
    ctx.obj["pypadre-app"].experiment_creator.set_param_values(experiment, parameters)


@experiment.command(name='get_params')
@click.option("--experiment", default=None, help='Name of the experiment from which parameters are to be retrieved')
@click.pass_context
def get_parameters(ctx, experiment):
    print(ctx.obj["pypadre-app"].experiment_creator.get_param_values(experiment))


@experiment.command(name="create_experiment")
@click.option('--name', default=None, help='Name of the experiment. If none UUID will be given')
@click.option('--description', default=None, help='Description of the experiment')
@click.option('--dataset', default=None, help='Name of the dataset to be used in the experiment')
@click.option('--workflow', default=None, help='Estimators to be used in the workflow')
@click.option('--backend', default=None, help='Backend of the experiment')
@click.pass_context
def create_experiment(ctx, name, description, dataset, workflow, backend):
    workflow_obj = None
    if workflow is not None:
        estimator_list = (workflow.replace(", ", ",")).split(sep=",")
        workflow_obj = ctx.obj["pypadre-app"].experiment_creator.create_test_pipeline(estimator_list)

    ctx.obj["pypadre-app"].experiment_creator.create(name, description, dataset, workflow_obj)


@experiment.command(name="run")
@click.pass_context
def execute(ctx):
    ctx.obj["pypadre-app"].experiment_creator.execute()


@experiment.command(name="do_experiments")
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
            datasets_list[idx] = ((datasets_list[idx].strip()).replace(", ", ",")).replace(" ,", ",")
            curr_exp_datasets = datasets_list[idx].split(sep=",")
            experiment_datasets_dict[experiments_list[idx]] = copy.deepcopy(curr_exp_datasets)

        ctx.obj["pypadre-app"].experiment_creator.do_experiments(experiment_datasets_dict)


@experiment.command(name="load_config_file")
@click.option('--filename', default=None, help='Path of the JSON file that contains the experiment parameters')
@click.pass_context
def load_config_file(ctx, filename):
    import os

    if os.path.exists(filename):
        ctx.obj["pypadre-app"].experiment_creator.parse_config_file(filename)
        ctx.obj["pypadre-app"].experiment_creator.execute()

    else:
        print('File does not exist')
