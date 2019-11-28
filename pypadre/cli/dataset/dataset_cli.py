"""
Command Line Interface for PADRE.

"""
import os
from ast import literal_eval

import click

from pypadre.core import plot
from pypadre.core.model.dataset.dataset import Dataset
#################################
####### DATASETS FUNCTIONS ##########
#################################
from pypadre.pod.app.dataset.dataset_app import DatasetApp


@click.group()
def dataset():
    """
    Commands for datasets.
    """


def _get_app(ctx) -> DatasetApp:
    return ctx.obj["pypadre-app"].datasets

def _print_table(ctx, *args, **kwargs):
    ctx.obj["pypadre-app"].print_tables(Dataset, *args, **kwargs)


@dataset.command(name="list")
@click.option('--columns', help='Show available column names', is_flag=True)
@click.option('--offset', '-o', default=0, help='Starting position of the retrieval')
@click.option('--limit', '-l', default=100, help='Number to retrieve')
@click.option('--search', '-s', default=None,
              help='Search dictonary.')# TODO further search instructions
@click.option('--column', '-c', help="Column to print", default=None, multiple=True)
@click.pass_context
def list(ctx, columns, search, offset, limit, column):

    if search:
        search = literal_eval(search)

    """Search for datasets."""
    if columns:
        print(Dataset.tablefy_columns())
        return 0
    _print_table(ctx, _get_app(ctx).list(search=search,offset=offset,size=limit),columns=column)


@dataset.command(name="get")
@click.argument('id' , type=click.STRING)
@click.option('--simple', '-s', help='Show only simple info', is_flag=True)
@click.pass_context
def get(ctx, id, simple=False):
    """Show dataset with the given id."""
    try:
        found = _get_app(ctx).get(id)
        if len(found) == 0:
            click.echo(click.style(str("No project found for id: " + id), fg="red"))
        elif len(found) >= 2:
            click.echo(click.style(str("Multiple projects found for id: " + id), fg="red"))
            _print_table(ctx, found)
        else:
            if simple:
                ctx.obj["pypadre-app"].print(found.pop())
            else:
                ctx.obj["pypadre-app"].print(found.pop().to_detail_string())
    except Exception as e:
        click.echo(click.style(str(e), fg="red"))


@dataset.command(name="sync")
@click.argument('dataset_id', default=None, required=False)
@click.option('--mode', '-m', help='Mode for the sync', type=click.STRING)
@click.pass_context
def sync(ctx, dataset_id, mode):
    """Synchronizes the backends."""
    _get_app(ctx).sync(name=dataset_id, mode=mode)


@dataset.command(name="load", context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
# @click.option('--binary/--no-binary', default=True, help='download binary'
@click.option('--defaults', '-d', help='Load defaults', is_flag=True)
@click.option('--source', '-s', help='Source for the download', type=click.STRING)
@click.option('--file', '-f', help='Source if you want to load a file', type=click.Path(exists=True))
@click.pass_context
def load(ctx, defaults, source=None, file=None):
    """Loads the dataset from given source."""

    arguments = dict()
    for item in ctx.args:
        arguments.update([item.split('=')])

    ds_app = _get_app(ctx)

    if file is not None:
        print(ds_app.load(file, **arguments))

    elif source is not None:
        print(ds_app.load(source, **arguments))

    if defaults:
        ds_app.load_defaults()


@dataset.command(name="scatter_plot")
@click.argument('id', type=click.STRING)
@click.option('--x', '-x', help='X attribute', type=click.STRING)
@click.option('--y', '-y', help='Y attribute', type=click.STRING)
@click.option('--name', '-n', default='scatter_plot.json', help='File name to be saved as json', type=click.STRING)
@click.pass_context
def scatter_plot(ctx, id, x, y, name):
    """Create scatter plot and save it on local system for given dataset.

    Example command:
        -> dataset scatter_plot _iris_dataset --x sepal\ length\ (cm) --y sepal\ width\ (cm)
    """
    try:
        app = _get_app(ctx)
        found = app.get(id)
        if len(found) == 0:
            click.echo(click.style(str("No dataset found for id: " + id), fg="red"))
        elif len(found) >= 2:
            click.echo(click.style(str("Multiple datasets found for id: " + id), fg="red"))
            _print_table(ctx, found)
        else:
            ds = found[0]
            plt = plot.DataPlot(ds)
            vis = plt.get_scatter_plot(x, y)
            app.service.backends[0].put_visualization(vis,
                                                      file_name=name,
                                                      base_path=os.path.join(app.service.backends[0].root_dir, id))

    except Exception as e:
        click.echo(click.style(str(e), fg="red"))


@dataset.command(name="class_balance")
@click.argument('id', type=click.STRING)
@click.option('--title', '-t', default='Class balance chart', help='Title of the chart', type=click.STRING)
@click.option('--name', '-n', default='class_balance.json', help='File name to be saved as json', type=click.STRING)
@click.pass_context
def class_balance(ctx, id, title, name):
    """Create class balance chart and save it on local system for given dataset.

    Example command:
        -> dataset class_balance _iris_dataset --title  Test\ title
    """
    try:
        app = _get_app(ctx)
        found = app.get(id)
        if len(found) == 0:
            click.echo(click.style(str("No dataset found for id: " + id), fg="red"))
        elif len(found) >= 2:
            click.echo(click.style(str("Multiple datasets found for id: " + id), fg="red"))
            _print_table(ctx, found)
        else:
            ds = found[0]
            plt = plot.DataPlot(ds)
            vis = plt.plot_class_balance(title)
            app.service.backends[0].put_visualization(vis,
                                                      file_name=name,
                                                      base_path=os.path.join(app.service.backends[0].root_dir, id))

    except Exception as e:
        click.echo(click.style(str(e), fg="red"))


@dataset.command(name="correlation_matrix")
@click.argument('id', type=click.STRING)
@click.option('--title', '-t', default='Correlation matrix chart', help='Title of the chart', type=click.STRING)
@click.option('--name', '-n', default='correlation_matrix.json', help='File name to be saved as json',
              type=click.STRING)
@click.pass_context
def correlation_matrix(ctx, id, title, name):
    """Create correlation matrix chart and save it on local system for given dataset.

    Example command:
        -> dataset correlation_matrix _iris_dataset
    """
    try:
        app = _get_app(ctx)
        found = app.get(id)
        if len(found) == 0:
            click.echo(click.style(str("No dataset found for id: " + id), fg="red"))
        elif len(found) >= 2:
            click.echo(click.style(str("Multiple datasets found for id: " + id), fg="red"))
            _print_table(ctx, found)
        else:
            ds = found[0]
            plt = plot.DataPlot(ds)
            vis = plt.plot_correlation_matrix(title)
            app.service.backends[0].put_visualization(vis,
                                                      file_name=name,
                                                      base_path=os.path.join(app.service.backends[0].root_dir, id))

    except Exception as e:
        click.echo(click.style(str(e), fg="red"))