"""
Command Line Interface for PADRE.

"""
import click


#################################
####### PROJECT FUNCTIONS ##########
#################################
from pypadre.app import PadreApp
from pypadre.core.model.project import Project


@click.group(name="project", invoke_without_command=True)
@click.pass_context
def project(ctx):
    """
    commands for projects
    """
    if ctx.invoked_subcommand is None:
        if ctx.obj["pypadre-app"].config.get("project", "DEFAULTS") is not None:
            click.echo('Current default project is ' + ctx.obj["pypadre-app"].config.get("project", "DEFAULTS"))


@project.command(name="select")
@click.argument('project_name')
@click.pass_context
def select(ctx, id):
    # Set as active project
    ctx.obj["pypadre-app"].config.set("project", id, "DEFAULTS")


@project.command(name="list")
@click.pass_context
def list(ctx):
    # List all the projects that are currently saved
    app: PadreApp = ctx.obj["pypadre-app"]
    # TODO fill search
    app.print_tables(app.projects.list())


@project.command(name="create")
@click.pass_context
def create(ctx):
    # Create a new project
    app: PadreApp = ctx.obj["pypadre-app"]
    # TODO fill project data
    p = Project()
    app.projects.put(p)
