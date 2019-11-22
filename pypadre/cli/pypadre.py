#!/usr/bin/env python

"""
Command Line Interface for PADRE.

"""
# todo support config file https://stackoverflow.com/questions/46358797/python-click-supply-arguments-and-options-from-a-configuration-file
import os

import click
from click_shell import shell

from pypadre.cli.computation import computation_cli
from pypadre.cli.execution import execution_cli
from pypadre.cli.run import run_cli
from pypadre.pod.app import PadreConfig
from pypadre.pod.app.padre_app import PadreAppFactory
from .config import config_cli
from .dataset import dataset_cli
from .experiment import experiment_cli
from .metric import metric_cli
from .project import project_cli


# # Make prompt changeable
# def get_prompt(self):
#     try:
#         return self.ctx.obj['prompt']
#     except KeyError:
#         return self.prompt
#
#
# ClickCmd.get_prompt = get_prompt


#################################
#######      MAIN      ##########
#################################
# @click.group()
@shell(prompt='pypadre > ', intro='Please insert your commands...', hist_file=os.path.join(os.path.expanduser('~'), '.click-pypadre-history'))
@click.option('--config-file', '-c',
              type=click.Path(),
              default=os.path.expanduser('~/.padre.cfg'),
              )
@click.pass_context
def pypadre(ctx, config_file):
    """
    Setup padre command line interface using the provided config file.

    Default config file: ~/.padre.cfg
    """

    click.echo("Initializing padre...")

    click.echo("""\
    
        N00K0K0KKKKKKKKKKKKKKKKKKKKKKKKKK0KKK00N
        KOXWWWWMWWWWMMWWWMNK0KXWWWWWWWWWWMMWMXOX
        KOXWWWWMMWWWWWMMWWK:.'dNWWWWWWWWWWMWMXOX
        KONMWWMMMMMWWMMWXNK, .,xXWWWWWWWMMMWMXOX
        KONMWWWMMMMWWMNx:kK: ...;OWWWWWWWMWWMXOX
        KkXMWWWMMMMMWWd..xKo.lOc.'0WWMMWWWWWMXOX
        KkKWWWMMMMWWWK; .dKo.':, .dWMWMWWWWWMXOX
        KkXWMWMMWMWWWK; .,:'.    .dWWWMWWMMWMXOX
        KkXWMMMMWWWWNXd.       ..,0MWWWWWMMWWKkX
        KkXWWMMMWWWWWNKc       .'xWWWWWWWWWWWKkX
        KOXNNNWMMWWWWNO;  ..... .oXWWWWWWMWWMXOX
        KkXWWNNWWWWWNk, .;c,.,;. .cXMMMWMMMMMXOX
        KOXWNWWNNNWWWKkc........cdONMMMMMMMMMXOX
        KkXWNNWWNNWWWNW0'      .OMMMMWMMMMMWMXOX
        Kx0NNXNWWWWMWWWk.'o,   .kWMMMWWMMMMMMXOX
        KdONNNWWWWWMWWWo.cKc....oNMMWWMMMMMMMXOX
        KkXNNWWWWWWWWW0, oXo,dd':0WWWWWWWMWWMXOX
        0x0XKKXXKXXXX0: .dXd:k0;'o0XXXXXXXXXX0kX
        0lldoloooodl;,. .;l;,d0d:ldlloooodoodldX
        0lldooooddd:...    ...,,'.:lcloooolodldX
        Klcoollol:'.       ';:dxxdxkxdolllcloldX
        0lcooooo;,,       ...';;,'...,oololloloX
        0lldoooo:'.   ...........   .,clodooocoX
        0lldoool;.    ........    .. .coldoooldX
        0lldolool:;,,''..........';,,:cloooooloX
        0lldoooooodooolllloooddooooolooolooodldX
        NK000000000000000000000000000000000000KW
        """)

    # load default configuration
    config = PadreConfig(config_file)
    # create context object
    ctx.obj = {
        'config-app': config,
        'pypadre-app': PadreAppFactory.get(config),
        'prompt': "pypadre > "
    }
    click.echo("Padre loaded!")

# @pypadre.command(name="authenticate")
# @click.option('--user', default=None, help='User on server')
# @click.option('--passwd', default=None, help='Password for given user')
# @click.pass_context
# def authenticate(ctx, user, passwd):
#     """
#     Get new authentication token and store it in the config. Authenticate with given credentials, in case credentials
#     are not provided default credentials will be used.
#     """
#     ctx.obj["pypadre-app"].authenticate(user, passwd)


pypadre.add_command(config_cli.config)
pypadre.add_command(dataset_cli.dataset)
pypadre.add_command(project_cli.project)
pypadre.add_command(experiment_cli.experiment)
pypadre.add_command(execution_cli.execution)
pypadre.add_command(run_cli.run)
pypadre.add_command(computation_cli.computation)
pypadre.add_command(metric_cli.metric)


if __name__ == '__main__':
    pypadre()
