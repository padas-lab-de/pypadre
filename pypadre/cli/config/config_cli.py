"""
Command Line Interface for PADRE.

"""
import click

from pypadre.app import PadreConfig
from pypadre.app.padre_app import PadreFactory


@click.group()
def config():
    """
    Commands for the configuration.

    Default config file: ~/.padre.cfg
    """


def _get_app(ctx) -> PadreConfig:
    return ctx.obj["config-app"]


@config.command(name="get")
@click.option('--param', default=None, help='Get value of given param')
@click.option('--section', default=None, help='Get value from given section')
@click.pass_context
def get(ctx, param, section):
    """
    commands for the configuration

    Default config file: ~/.padre.cfg
    """
    if section is None:
        result = _get_app(ctx).get(param)
    else:
        result = _get_app(ctx).get(param, section=section)
    print(result)


@config.command(name="set")
@click.option('--param', nargs=2, default=None, help='key value pair as tuple')
@click.option('--section', default=None, help='Set key, value for given section')
@click.pass_context
def set_config_param(ctx, param, section):
    """
    Sets key, value in config. param must be a tuple
    """
    if section is None:
        _get_app(ctx).set(param[0], param[1])
    else:
        _get_app(ctx).set(param[0], param[1], section)
    _get_app(ctx).save()

    # Reinitialize app with changed configuration
    ctx.obj['pypadre-app'] = PadreFactory.get(_get_app(ctx))


@config.command(name="list")
@click.option('--section', default=None, help='Get list of params from given section')
@click.pass_context
def list_config_params(ctx, section):
    """
    List all values in config
    """
    if section is None:
        result = _get_app(ctx).config['GENERAL'].keys()
    else:
        result = _get_app(ctx).config[section].keys()
    print(result)
