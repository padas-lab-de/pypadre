"""
Command Line Interface for PADRE.

"""
from typing import cast

import click


#################################
####### DATASETS FUNCTIONS ##########
#################################
from pypadre.app.dataset.dataset_app import DatasetApp
from pypadre.enums.data_sources import DataSources


@click.group()
def dataset():
    """
    commands for the configuration

    Default config file: ~/.padre.cfg
    """


@dataset.command(name="list")
# @click.option('--start', default=0, help='start number of the dataset')
# @click.option('--count', default=999999999, help='Number of datasets to retrieve')
@click.option('--search', default=None,
              help='search string')
@dataset.pass_context
def datasets(ctx, search):
    """list all available datasets"""
    # TODO like pageable (sort, offset etc.)
    cast(DatasetApp, ctx.obj["pypadre"].datasets).list(search=search)


@dataset.command(name="get")
@click.argument('dataset_id')
# @click.option('--binary/--no-binary', default=True, help='download binary')
# @click.option('--format', '-f', default="numpy", help='format for binary download')
@click.pass_context
def dataset(ctx, dataset_id):
    """downloads the dataset with the given id. id can be either a number or a valid url"""
    cast(DatasetApp, ctx.obj["pypadre"].datasets).get(dataset_id)


@dataset.command(name="load")
# @click.option('--binary/--no-binary', default=True, help='download binary')
@click.option('--source', '-s', default="sklearn", help='Source for the download', type=click.STRING)
@click.option('--file', '-s', help='Source for the download', type=click.Path(exists=True))
@click.argument(help='Config for loading', nargs=-1, type=click.Tuple([str,str]))
@click.pass_context
def dataset(ctx, source=None, file=None):
    """downloads the dataset with the given id."""
    ds_app = cast(DatasetApp, ctx.obj["pypadre"].datasets)
    if file is not None:
        ds_app.load(file)

    if source is not None:
        # TODO

# @dataset.command(name="import")
# @click.option('--sklearn/--no-sklearn', default=True, help='import sklearn default datasets')
# @click.pass_context
# def do_import(ctx, sklearn):
#     """import default datasets from different sources specified by the flags"""
#     ctx.obj["pypadre"].datasets.do_default_imports(sklearn)


# @dataset.command(name="import")
# @click.argument('dataset_id')
# @click.option('--source', required=True, multiple='true', help='source of the datasets',
#               type=click.Choice(DataSources))
# @click.pass_context
# def dataset(ctx, dataset_id, source):
#     """downloads the dataset with the given id. id can be either a number or a valid url"""
#     if source == DataSources.oml:
#         ctx.obj["pypadre"].datasets.get_openml_dataset(dataset_id)
#     if source == DataSources.sklearn:
#         ctx.obj["pypadre"].datasets.do_default_imports(True)


# @dataset.command(name="upload_scratchdata_multi")
# @click.pass_context
# def dataset(ctx):
#     """uploads sklearn toy-datasets, top 100 Datasets form OpenML and some Graphs
#     Takes about 1 hour
#     Requires about 7GB of Ram"""
#
#     auth_token=ctx.obj["pypadre"].config.get("token")
#
#     ctx.obj["pypadre"].datasets.upload_scratchdatasets(auth_token,max_threads=8,upload_graphs=True)
#
#
# @dataset.command(name="upload_scratchdata_single")
# @click.pass_context
# def dataset(ctx):
#     """uploads sklearn toy-datasets, top 100 Datasets form OpenML and some Graphs
#     Takes several hours
#     Requires up to 7GB of Ram"""
#
#     auth_token = ctx.obj["pypadre"].config.get("token")
#
#     ctx.obj["pypadre"].datasets.upload_scratchdatasets(auth_token, max_threads=1, upload_graphs=True)