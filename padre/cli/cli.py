"""
Command Line Interface for PADRE.

"""
#todo support config file https://stackoverflow.com/questions/46358797/python-click-supply-arguments-and-options-from-a-configuration-file

import click
import os
import configparser

#################################
####### DEFAULT VALUES ##########
#################################
from padre.backend.http import PadreHTTPClient
from padre.ds_import import load_sklearn_toys

_BASE_URL = "http://localhost:8080/api"

def padre_client_from_ctx(ctx):
    return PadreHTTPClient(**ctx)

def set_default_config(config):
    config["API"] = {
        "base_url": _BASE_URL,
        "user": "",
        "passwd": ""
    }

#################################
#######      MAIN      ##########
#################################
@click.group()
@click.option(
    '--config-file', '-c',
    type=click.Path(),
    default='~/.padre.cfg',
)
@click.option('--base-url', '-b', envvar="PADRE_URL",
              type=str,
              help="Base url for the PADRE REST API")
@click.pass_context
def main(ctx, config_file, base_url):
    """
    padre command line interface.

    Default config file: ~/.padre.cfg
    """
    filename = os.path.expanduser(config_file)
    config = configparser.ConfigParser()
    if os.path.exists(filename):
            config.read(filename)
    else:
        set_default_config(config)

    ctx.obj = {
        'config-file': config_file,
        'API': config["API"]
    }

@main.command()
@click.option('--reset/--no-reset', default=False,
              help='reset the configuration to the default values or create a new configuration file')
@click.pass_context
def config(ctx, reset):
    """
    Store configuration values in a file.
    """
    filename = os.path.expanduser(ctx.obj["config-file"])
    config = configparser.ConfigParser()
    if not reset:
        for k, v in ctx.obj.items():
            config[k] = v
    else:
        set_default_config(config)
    with open(filename, "w") as cfile:
        config.write(cfile)

#################################
####### DATASETS FUNCTIONS ##########
#################################
@main.command(name="datasets")
@click.option('--start', default=0, help='start number of the dataset')
@click.option('--count', default=999999999, help='Number of datasets to retrieve')
@click.option('--search', default=None,
              help='search string')
@click.option('--all', '-a', default=False,
              help='show all fields available')
@click.option('--max', '-m', default=40,
              help='maximum number of characters to display per field')
@click.pass_context
def datasets(ctx, start, count, search, all, max):
    """list all available datasets"""
    client = padre_client_from_ctx(ctx.obj["API"])
    datasets = client.list_datasets(start, count, search)
    if not all:
        headings = ["  ID  ", "    Name     ", "      Type          ", "         #att        ", "      Created    "]
    else:
        headings = ["  ID  ", "ALL FIELDS (TODO: Make good heading"]

    click.echo("\t".join(headings))
    for ds in datasets:
        if not all:
            click.echo("\t".join([str(x)[:max] for x in [ds.id, ds.name, ds.type, ds.num_attributes, ds.metadata["createdAt"]]]))
        else:
            click.echo("\t".join([k + "=" + str(v)[:max] for k, v in ds.metadata.items()]))


@main.command(name="import")
@click.option('--sklearn/--no-sklearn',  default=True, help='import sklearn default datasets')
@click.pass_context
def imports(ctx, sklearn):
    """import default datasets from different sources specified by the flags"""
    client = padre_client_from_ctx(ctx.obj["API"])
    if sklearn:
        for ds in load_sklearn_toys():
            click.echo("Uploading dataset %s, %s, %s" % (ds.name, str(ds.size), ds.type))
            client.upload_dataset(ds, True)


@main.command(name="dataset")
@click.argument('dataset_id')
@click.option('--binary/--no-binary',  default=True, help='download binary')
@click.option('--format', '-f', default="numpy", help='format for binary download')
@click.pass_context
def dataset(ctx, dataset_id, binary, format):
    """downloads the dataset with the given id. id can be either a number or a valid url"""
    client = padre_client_from_ctx(ctx.obj["API"])
    ds = client.get_dataset(dataset_id, binary, format=format)
    click.echo(f"Metadata for dataset {ds.id}")
    for k, v in ds.metadata.items():
        click.echo("\t%s=%s" % (k, str(v)))
    click.echo("Available formats:")
    formats = client.get_dataset_formats(dataset_id)
    for f in formats:
        click.echo("\t%s" % (f))
    click.echo("Binary description:")
    for k, v in ds.describe().items():
        click.echo("\t%s=%s" % (k, str(v)))


if __name__ == '__main__':
    main()

