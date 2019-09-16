"""
Command Line Interface for PADRE.

"""
import click
#################################
####### PROJECT FUNCTIONS ##########
#################################
from click_shell import make_click_shell

from pypadre.app.project.project_app import ProjectApp
from pypadre.cli.experiment import experiment_cli
from pypadre.core.model.project import Project
from pypadre.pod.validation.json_schema import JsonSchemaRequiredHandler
from pypadre.pod.validation.validation import ValidateableFactory


@click.group(name="project", invoke_without_command=True)
@click.pass_context
def project(ctx):
    """
    Commands for projects.
    """

    #if ctx.invoked_subcommand is None:
    #    if ctx.obj["pypadre-app"].config.get("project", "DEFAULTS") is not None:
    #        click.echo('Current default project is ' + ctx.obj["pypadre-app"].config.get("project", "DEFAULTS"))


def _get_app(ctx) -> ProjectApp:
    return ctx.obj["pypadre-app"].projects


def _print_table(ctx, *args, **kwargs) -> ProjectApp:
    print(ctx.obj["pypadre-app"].print_tables(*args, **kwargs))


@project.command(name="list")
@click.option('--offset', '-o', default=0, help='start number of the dataset')
@click.option('--limit', '-l', default=100, help='Number of datasets to retrieve')
@click.option('--search', '-s', default=None,
              help='search string')
@click.option('--column', '-c', help="Column to print", default=None, multiple=True)
@click.pass_context
def list(ctx, search, offset, limit, column):
    """
    List projects defined in the padre environment
    """
    # List all the projects that are currently saved
    _print_table(ctx, _get_app(ctx).list(search=search, offset=offset, size=limit), columns=column)


@project.command(name="create")
@click.pass_context
def create(ctx):
    """
    Create a new project
    """
    # Create a new project
    def get_value(obj, e, options):
        return click.prompt(e.message + '. Please enter a value', type=str)

    p = ValidateableFactory.make(Project, handlers=[JsonSchemaRequiredHandler(validator="required", get_value=get_value)])
    _get_app(ctx).put(p)


@click.group(name="select", invoke_without_command=True)
@click.argument('name', type=click.STRING)
@click.pass_context
def select(ctx, name):
    """
    Select a project as active
    """
    # Set as active project
    projects = _get_app(ctx).list({"name": name})
    if len(projects) == 0:
        print("Project {0} not found!".format(name))
        return -1
    if len(projects) > 1:
        print("Multiple matching projects found!")
        _print_table(ctx, projects)
        return -1
    s = make_click_shell(ctx, prompt='pypadre > p: ' + name + ' > ', intro='Selecting project ' + name)
    ctx.obj['project'] = projects.pop(0)
    s.cmdloop()
    ctx.obj['project'] = None


project.add_command(select)
select.add_command(experiment_cli.experiment)
