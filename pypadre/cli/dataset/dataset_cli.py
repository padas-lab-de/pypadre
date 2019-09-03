"""
Command Line Interface for PADRE.

"""
from typing import cast

import click

#################################
####### DATASETS FUNCTIONS ##########
#################################
from pypadre.app.dataset.dataset_app import DatasetApp
from pypadre.core.model.dataset.dataset import Dataset
from pypadre.printing.tablefyable import registry
from pypadre.printing.util.print_util import to_table


@click.group()
def dataset():
    """
    commands for the data sets
    """


def _get_app(ctx) -> DatasetApp:
    return ctx.obj["pypadre-app"].datasets


@dataset.command(name="list")
@click.option('--columns', help='Show only columns which are available', is_flag=True)
@click.option('--offset', '-o', default=0, help='start number of the dataset')
@click.option('--limit', '-l', default=100, help='Number of datasets to retrieve')
@click.option('--search', '-s', default=None,
              help='search string')
@click.option('--column', '-c', help="Column to print", default=None, multiple=True)
@click.pass_context
def list(ctx, columns, search, offset, limit, column):
    if columns:
        print(Dataset.tablefy_columns())
        return 0
    """list all available datasets"""
    # TODO like pageable (sort, offset etc.)
    print(to_table(_get_app(ctx).list(search=search, offset=offset, size=limit),
          columns=column))


@dataset.command(name="get")
@click.argument('dataset_id')
@click.option('--simple', '-s', help='Show only simple info', is_flag=True)
@click.pass_context
def get(ctx, dataset_id, simple=False):
    """downloads the dataset with the given id. id can be either a number or a valid url"""
    if simple:
        print('\n'.join(map(str, _get_app(ctx).get(dataset_id))))
    else:
        print('\n'.join([d.to_detail_string() for d in _get_app(ctx).get(dataset_id)]))


@dataset.command(name="sync")
@click.argument('dataset_id', default=None, required=False)
@click.option('--mode', '-m', help='Mode for the sync', type=click.STRING)
@click.pass_context
def sync(ctx, dataset_id, mode):
    """downloads the dataset with the given id. id can be either a number or a valid url"""
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
    """downloads the dataset with the given id."""

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
