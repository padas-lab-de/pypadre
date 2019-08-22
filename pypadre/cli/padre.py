#!/usr/bin/env python

"""
Command Line Interface for PADRE.

"""
# todo support config file https://stackoverflow.com/questions/46358797/python-click-supply-arguments-and-options-from-a-configuration-file
import os
import click
from click_shell import shell

from pypadre.app import p_app, PadreApp, PadreConfig
from pypadre.app.padre_app import PadreFactory
from .config import config_cli
from .dataset import dataset_cli
from .project import project_cli
from .experiment import experiment_cli
from .metric import metric_cli


#################################
#######      MAIN      ##########
#################################
# @click.group()
@shell(prompt='padre > ', intro='Starting padre ...')
@click.option('--config-file', '-c',
              type=click.Path(),
              default=os.path.expanduser('~/.padre.cfg'),
              )
@click.option('--base-url', '-b', envvar="PADRE_URL",
              type=str,
              help="Base url for the PADRE REST API")
@click.pass_context
def pypadre_cli(ctx, config_file, base_url):
    """
    setup padre command line interface using the provided config file

    Default config file: ~/.padre.cfg
    """
    # load default config
    config = PadreConfig(config_file)
    # override defaults
    if base_url is not None:
        config["HTTP BACKEND"]["base_url"] = base_url
    # create context object
    ctx.obj = {
        'config-file': config_file,
        'pypadre': PadreFactory.get(config)
    }


@pypadre_cli.command(name="authenticate")
@click.option('--user', default=None, help='User on server')
@click.option('--passwd', default=None, help='Password for given user')
@click.pass_context
def authenticate(ctx, user, passwd):
    """
    To generate new token in config. Authenticate with given credentials, in case credentials
    are not provided default credentials will be used.
    """
    ctx.obj["pypadre"].authenticate(user, passwd)


pypadre_cli.add_command(config_cli)
pypadre_cli.add_command(dataset_cli)
pypadre_cli.add_command(project_cli)
pypadre_cli.add_command(experiment_cli)
pypadre_cli.add_command(metric_cli)

if __name__ == '__main__':
    pypadre_cli()
