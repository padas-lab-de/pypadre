"""
Command Line Interface for PADRE.

"""
from typing import cast

import click

#################################
####### DATASETS FUNCTIONS ##########
#################################
from pypadre.app.dataset.dataset_app import DatasetApp
from pypadre.printing.util.print_util import to_table


@click.group()
def dataset():
    """
    commands for the configuration

    Default config file: ~/.padre.cfg
    """


@dataset.command(name="list")
@click.option('--offset', default=0, help='start number of the dataset')
@click.option('--size', default=100, help='Number of datasets to retrieve')
@click.option('--search', default=None,
              help='search string')
@click.option('--column', help="Column to print", default=None, multiple=True)
@click.pass_context
def find(ctx, search, offset, size, column):
    """list all available datasets"""
    # TODO like pageable (sort, offset etc.)
    print(to_table(list(cast(DatasetApp, ctx.obj["pypadre-app"].datasets).list(search=search, offset=offset, size=size))),
          *column)


@dataset.command(name="get")
@click.argument('dataset_id')
@click.pass_context
def get(ctx, dataset_id):
    """downloads the dataset with the given id. id can be either a number or a valid url"""
    print(cast(DatasetApp, ctx.obj["pypadre-app"].datasets).get(dataset_id))


@dataset.command(name="sync")
@click.argument('dataset_id', default=None, required=False)
@click.option('--mode', '-m', help='Mode for the sync', type=click.STRING)
@click.pass_context
def sync(ctx, dataset_id, mode):
    """downloads the dataset with the given id. id can be either a number or a valid url"""
    cast(DatasetApp, ctx.obj["pypadre-app"].datasets).sync(name=dataset_id, mode=mode)


@dataset.command(name="load", context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
# @click.option('--binary/--no-binary', default=True, help='download binary')
@click.option('--source', '-s', default="sklearn", help='Source for the download', type=click.STRING)
@click.option('--file', '-s', help='Source for the download', type=click.Path(exists=True))
@click.pass_context
def dataset(ctx, source=None, file=None):
    """downloads the dataset with the given id."""

    arguments = dict()
    for item in ctx.args:
        arguments.update([item.split('=')])

    ds_app = cast(DatasetApp, ctx.obj["pypadre-app"].datasets)
    if file is not None:
        print(ds_app.load(file, **arguments))

    if source is not None:
        print(ds_app.load(source, **arguments))