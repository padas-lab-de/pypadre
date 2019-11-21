"""
Command Line Interface for PADRE.

"""
import os

import click
#################################
####### EXECUTION FUNCTIONS ##########
#################################
from click_shell import make_click_shell

from pypadre.cli.computation import computation_cli
from pypadre.cli.metric import metric_cli
from pypadre.cli.run import run_cli
from pypadre.core.model.execution import Execution
from pypadre.pod.app.project.execution_app import ExecutionApp


@click.group(name="execution")
@click.pass_context
def execution(ctx):
    pass


def _get_app(ctx) -> ExecutionApp:
    return ctx.obj["pypadre-app"].executions


def _print_table(ctx, *args, **kwargs):
    ctx.obj["pypadre-app"].print_tables(Execution, *args, **kwargs)


def _filter_selection(ctx, found):
    # filter for experiment selection
    if 'experiment' in ctx.obj:
        found = [f for f in found if f.parent == ctx.obj['experiment']]

    # filter for project selection
    elif 'project' in ctx.obj:
        found = [f for f in found if f.parent.parent == ctx.obj['project']]
    return found


@execution.command(name="list")
@click.option('--offset', '-o', default=0, help='start number of the dataset')
@click.option('--limit', '-l', default=100, help='Number of datasets to retrieve')
@click.option('--search', '-s', default=None,
              help='search string')
@click.option('--column', '-c', help="Column to print", default=None, multiple=True)
@click.pass_context
def list(ctx, search, offset, limit, column):
    """
    List executions defined in the padre environment
    """
    # List all the executions that are currently saved
    _print_table(ctx, _get_app(ctx).list(search=search, offset=offset, size=limit), columns=column)


@execution.command(name="get")
@click.argument('id', type=click.STRING)
@click.pass_context
def get(ctx, id):
    try:
        found = _filter_selection(ctx, _get_app(ctx).get(id))

        if len(found) == 0:
            click.echo(click.style(str("No execution found for id: " + id), fg="red"))
        elif len(found) >= 2:
            click.echo(click.style(str("Multiple executions found for id: " + id), fg="red"))
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
    Select a execution as active
    """
    # Set as active execution
    executions = _get_app(ctx).list({"id": id})
    if len(executions) == 0:
        print("Execution {0} not found!".format(id))
        return -1
    if len(executions) > 1:
        print("Multiple matching executions found!")
        _print_table(ctx, executions)
        return -1
    prompt = ctx.obj['prompt']
    s = make_click_shell(ctx, prompt=prompt + 'exe: ' + id + ' > ', intro='Selecting execution ' + id, hist_file=os.path.join(os.path.expanduser('~'), '.click-pypadre-history'))
    ctx.obj['prompt'] = prompt
    ctx.obj['execution'] = executions.pop(0)
    s.cmdloop()
    del ctx.obj['execution']


execution.add_command(select)
select.add_command(run_cli.run)
select.add_command(computation_cli.computation)
select.add_command(metric_cli.metric)
