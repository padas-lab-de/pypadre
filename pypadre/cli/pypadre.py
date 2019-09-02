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
def pypadre(ctx, config_file, base_url):
    """
    setup padre command line interface using the provided config file

    Default config file: ~/.padre.cfg
    """
    # load default config
    config = PadreConfig(config_file)
    # create context object
    ctx.obj = {
        'config-app': config,
        'pypadre-app': PadreFactory.get(config)
    }


@pypadre.command(name="authenticate")
@click.option('--user', default=None, help='User on server')
@click.option('--passwd', default=None, help='Password for given user')
@click.pass_context
def authenticate(ctx, user, passwd):
    """
    To generate new token in config. Authenticate with given credentials, in case credentials
    are not provided default credentials will be used.
    """
    ctx.obj["pypadre-app"].authenticate(user, passwd)


pypadre.add_command(config_cli.config)
pypadre.add_command(dataset_cli.dataset)
pypadre.add_command(project_cli.project)
pypadre.add_command(experiment_cli.experiment)
pypadre.add_command(metric_cli.metric)

if __name__ == '__main__':
    pypadre()
