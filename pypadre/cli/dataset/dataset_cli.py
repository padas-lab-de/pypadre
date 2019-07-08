"""
Command Line Interface for PADRE.

"""
import click


#################################
####### DATASETS FUNCTIONS ##########
#################################
from pypadre.enums import data_sources


@click.group()
def dataset():
    """
    commands for the configuration

    Default config file: ~/.padre.cfg
    """


@dataset.command(name="list")
@click.option('--start', default=0, help='start number of the dataset')
@click.option('--count', default=999999999, help='Number of datasets to retrieve')
@click.option('--search', default=None,
              help='search string')
@dataset.pass_context
def datasets(ctx, start, count, search):
    """list all available datasets"""
    ctx.obj["pypadre"].datasets.list_datasets(start, count, search)


@dataset.command(name="import")
@click.option('--sklearn/--no-sklearn', default=True, help='import sklearn default datasets')
@click.pass_context
def do_import(ctx, sklearn):
    """import default datasets from different sources specified by the flags"""
    ctx.obj["pypadre"].datasets.do_default_imports(sklearn)


@dataset.command(name="get")
@click.argument('dataset_id')
@click.option('--binary/--no-binary', default=True, help='download binary')
@click.option('--format', '-f', default="numpy", help='format for binary download')
@click.pass_context
def dataset(ctx, dataset_id, binary, format):
    """downloads the dataset with the given id. id can be either a number or a valid url"""
    ctx.obj["pypadre"].datasets.get_dataset(dataset_id, binary, format)


@dataset.command(name="import")
@click.argument('dataset_id')
@click.option('--source', required=True, multiple='true', help='source of the datasets',
              type=click.Choice(data_sources))
@click.pass_context
def dataset(ctx, dataset_id, source):
    """downloads the dataset with the given id. id can be either a number or a valid url"""
    if source == "oml":
        ctx.obj["pypadre"].datasets.get_openml_dataset(dataset_id)


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