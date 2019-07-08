"""
Command Line Interface for PADRE.

"""
import click

@click.group()
def config():
    """
    commands for the configuration

    Default config file: ~/.padre.cfg
    """

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
        result = ctx.obj["pypadre"].config.get(param)
    else:
        result = ctx.obj["pypadre"].config.get(param, section)
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
        ctx.obj["pypadre"].config.set(param[0], param[1])
    else:
        ctx.obj["pypadre"].config.set(param[0], param[1], section)
    ctx.obj["pypadre"].config.save()


@config.command(name="list")
@click.option('--section', default=None, help='Get list of params from given section')
@click.pass_context
def list_config_params(ctx, section):
    """
    List all values in config
    """
    result = ctx.obj["pypadre"].config.config[section].keys()
    print(result)
