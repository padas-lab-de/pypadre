"""
Command Line Interface for PADRE.

"""

import click

from pypadre.core.model.computation.computation import Computation
from pypadre.pod.app.project.computation_app import ComputationApp


#################################
####### COMPUTATION FUNCTIONS ##########
#################################


@click.group(name="computation")
@click.pass_context
def computation(ctx):
    pass


def _get_app(ctx) -> ComputationApp:
    return ctx.obj["pypadre-app"].computations


def _print_table(ctx, *args, **kwargs):
    ctx.obj["pypadre-app"].print_tables(Computation, *args, **kwargs)


def _filter_selection(ctx, found):
    # filter for run selection
    if 'run' in ctx.obj:
        found = [f for f in found if f.parent == ctx.obj['run']]

    # filter for execution selection
    if 'execution' in ctx.obj:
        found = [f for f in found if f.parent.parent == ctx.obj['execution']]

    # filter for experiment selection
    elif 'experiment' in ctx.obj:
        found = [f for f in found if f.parent.parent.parent == ctx.obj['experiment']]

    # filter for project selection
    elif 'project' in ctx.obj:
        found = [f for f in found if f.parent.parent.parent.parent == ctx.obj['project']]
    return found


@computation.command(name="list")
@click.option('--offset', '-o', default=0, help='start number of the dataset')
@click.option('--limit', '-l', default=100, help='Number of datasets to retrieve')
@click.option('--search', '-s', default=None,
              help='search string')
@click.option('--column', '-c', help="Column to print", default=None, multiple=True)
@click.pass_context
def list(ctx, search, offset, limit, column):
    """
    List computations defined in the padre environment
    """
    # List all the computations that are currently saved
    _print_table(ctx, _get_app(ctx).list(search=search, offset=offset, size=limit), columns=column)


@computation.command(name="get")
@click.argument('id', type=click.STRING)
@click.pass_context
def get(ctx, id):
    try:
        found = _filter_selection(ctx, _get_app(ctx).get(id))
        if len(found) == 0:
            click.echo(click.style(str("No computation found for id: " + id), fg="red"))
        elif len(found) >= 2:
            click.echo(click.style(str("Multiple computations found for id: " + id), fg="red"))
            _print_table(ctx, found)
        else:
            ctx.obj["pypadre-app"].print(found.pop())
    except Exception as e:
        click.echo(click.style(str(e), fg="red"))

