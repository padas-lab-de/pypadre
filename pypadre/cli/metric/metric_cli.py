"""
Command Line Interface for PADRE.

"""
import click

#################################
####### METRICS FUNCTIONS ##########
#################################
@click.group(name="metric")
@click.option('-e', '--experiment')
def metric(ctx, experiment):
    """
    Commands for the metrics.
    """
    ctx.obj["experiment"] = experiment


@metric.command(name="compare_metrics")
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
    ctx.obj["pypadre-app"].metrics_evaluator.read_run_directories(dir_path_list)
    ctx.obj["pypadre-app"].metrics_evaluator.get_unique_estimators_parameter_names()
    ctx.obj["pypadre-app"].metrics_evaluator.read_split_metrics()
    ctx.obj["pypadre-app"].metrics_evaluator.analyze_runs(estimators_list, metrics_list)
    print(ctx.obj["pypadre-app"].metrics_evaluator.display_results())


@metric.command(name="reevaluate_metrics")
@click.option('--path', default=None, help='Path of experiments whose metrics are to be reevaluated')
@click.pass_context
def reevaluate_runs(ctx, path):
    path_list = (path.replace(", ", ",")).split(sep=",")
    ctx.obj["pypadre-app"].metrics_reevaluator.get_split_directories(dir_path=path_list)
    ctx.obj["pypadre-app"].metrics_reevaluator.recompute_metrics()


@metric.command(name="list_experiments")
@click.pass_context
def list_experiments(ctx):
    print(ctx.obj["pypadre-app"].metrics_evaluator.get_experiment_directores())


@metric.command(name='get_available_estimators')
@click.pass_context
def get_available_estimators(ctx):
    estimators = ctx.obj["pypadre-app"].metrics_evaluator.get_unique_estimator_names()
    if estimators is None:
        print('No estimators found for the runs compared')
    else:
        print(estimators)


@metric.command(name="list_estimator_params")
@click.option('--estimator_name', default=None, help='Params of this estimator will be displayed')
@click.option('--selected_params', default=None, help='List the values of only these parameters')
@click.option('--return_all_values', default=None,
              help='Returns all the parameter values for the estimator including the default values')
def list_estimator_params(ctx, estimator_name, selected_params, return_all_values):
    params_list = \
        ctx.obj["pypadre-app"].metrics_evaluator.get_estimator_param_values\
            (estimator_name, selected_params, return_all_values)
    if params_list is None:
        print('No parameters obtained')
    else:
        for param in params_list:
            print(param, params_list.get(param, '-'))
