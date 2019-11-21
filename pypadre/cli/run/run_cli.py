"""
Command Line Interface for PADRE.

"""
import os

import click
#################################
####### RUN FUNCTIONS ##########
#################################
from click_shell import make_click_shell

from pypadre.cli.computation import computation_cli
from pypadre.cli.metric import metric_cli
from pypadre.core.model.computation.run import Run
from pypadre.pod.app.project.run_app import RunApp


@click.group(name="run")
@click.pass_context
def run(ctx):
    pass


def _get_app(ctx) -> RunApp:
    return ctx.obj["pypadre-app"].runs


def _print_table(ctx, *args, **kwargs):
    ctx.obj["pypadre-app"].print_tables(Run, *args, **kwargs)


def _filter_selection(ctx, found):
    # filter for execution selection
    if 'execution' in ctx.obj:
        found = [f for f in found if f.parent.id == ctx.obj['execution']]

    # filter for experiment selection
    elif 'experiment' in ctx.obj:
        found = [f for f in found if f.parent.parent.id == ctx.obj['experiment']]

    # filter for project selection
    elif 'project' in ctx.obj:
        found = [f for f in found if f.parent.parent.parent.id == ctx.obj['project']]
    return found


@run.command(name="list")
@click.option('--offset', '-o', default=0, help='start number of the dataset')
@click.option('--limit', '-l', default=100, help='Number of datasets to retrieve')
@click.option('--search', '-s', default=None,
              help='search string')
@click.option('--column', '-c', help="Column to print", default=None, multiple=True)
@click.pass_context
def list(ctx, search, offset, limit, column):
    """
    List runs defined in the padre environment
    """
    # List all the runs that are currently saved
    _print_table(ctx, _get_app(ctx).list(search=search, offset=offset, size=limit), columns=column)


@run.command(name="get")
@click.argument('id', type=click.STRING)
@click.pass_context
def get(ctx, id):
    try:
        found = _filter_selection(ctx, _get_app(ctx).get(id))
        if len(found) == 0:
            click.echo(click.style(str("No run found for id: " + id), fg="red"))
        elif len(found) >= 2:
            click.echo(click.style(str("Multiple runs found for id: " + id), fg="red"))
            _print_table(ctx, found)
        else:
            ctx.obj["pypadre-app"].print(found.pop())
    except Exception as e:
        click.echo(click.style(str(e), fg="red"))


@click.group(name="select", invoke_without_command=True)
@click.argument('id', type=click.STRING)
@click.pass_context
def select(ctx, id):
    """
    Select a run as active
    """
    # Set as active run
    runs = _get_app(ctx).list({"id": id})
    if len(runs) == 0:
        print("Run {0} not found!".format(id))
        return -1
    if len(runs) > 1:
        print("Multiple matching runs found!")
        _print_table(ctx, runs)
        return -1
    prompt = ctx.obj['promp']
    s = make_click_shell(ctx, prompt=prompt + 'run: ' + id + ' > ', intro='Selecting run ' + id, hist_file=os.path.join(os.path.expanduser('~'), '.click-pypadre-history'))
    ctx.obj['promp'] = prompt
    ctx.obj['run'] = runs.pop(0)
    s.cmdloop()
    del ctx.obj['run']


run.add_command(select)
select.add_command(computation_cli.computation)
select.add_command(metric_cli.metric)
