"""
Command Line Interface for PADRE.

"""
#todo support config file https://stackoverflow.com/questions/46358797/python-click-supply-arguments-and-options-from-a-configuration-file
import click
import padre.app.padre_app as app
#################################
#######      MAIN      ##########
#################################
@click.group()
@click.option('--config-file', '-c',
    type=click.Path(),
    default='~/.padre.cfg',
)
@click.option('--base-url', '-b', envvar="PADRE_URL",
              type=str,
              help="Base url for the PADRE REST API")
@click.pass_context
def pypadre_cli(ctx, config_file, base_url):
    """
    padre command line interface.

    Default config file: ~/.padre.cfg
    """
    # load default config
    config = app.default_config.copy()
    # override defaults
    if base_url is not None:
        config["HTTP"]["base_url"] = base_url
    # create app objects
    _http = app.padre_http_from_config(config)
    _file = app.padre_filecache_from_config(config)
    # create context object
    ctx.obj = {
        'config-file': config_file,
        'API': config["HTTP"],
        'http-client': _http,
        'pypadre': app.PadreApp(_http, _file, click.echo)
    }

#################################
####### DATASETS FUNCTIONS ##########
#################################
@pypadre_cli.command(name="datasets")
@click.option('--start', default=0, help='start number of the dataset')
@click.option('--count', default=999999999, help='Number of datasets to retrieve')
@click.option('--search', default=None,
              help='search string')
@click.pass_context
def datasets(ctx, start, count, search):
    """list all available datasets"""
    ctx.obj["pypadre"].datasets.list(start, count, search)

@pypadre_cli.command(name="import")
@click.option('--sklearn/--no-sklearn',  default=True, help='import sklearn default datasets')
@click.pass_context
def do_import(ctx, sklearn):
    """import default datasets from different sources specified by the flags"""
    ctx.obj["pypadre"].datasets.do_default_imports(sklearn)

@pypadre_cli.command(name="dataset")
@click.argument('dataset_id')
@click.option('--binary/--no-binary',  default=True, help='download binary')
@click.option('--format', '-f', default="numpy", help='format for binary download')
@click.pass_context
def dataset(ctx, dataset_id, binary, format):
    """downloads the dataset with the given id. id can be either a number or a valid url"""
    ctx.obj["pypadre"].datasets.get(dataset_id, binary, format)


if __name__ == '__main__':
    pypadre_cli()

