"""
Command Line Interface for PADRE.

"""
import click


#################################
####### PROJECT FUNCTIONS ##########
#################################

@click.group(name="project", invoke_without_command=True)
@click.pass_context
def project(ctx):
    """
    commands for projects
    """
    if ctx.invoked_subcommand is None:
        if ctx.obj["pypadre"].config.get("project", "DEFAULTS") is not None:
            click.echo('Current default project is ' + ctx.obj["pypadre"].config.get("project", "DEFAULTS"))


@project.command(name="select")
@click.argument('project_name')
@click.pass_context
def select(ctx, id):
    # Set as active project
    ctx.obj["pypadre"].config.set("project", id, "DEFAULTS")


@project.command(name="list")
@click.pass_context
def list(ctx):
    # List all the projects that are currently saved
    # TODO
    ctx


@project.command(name="create")
@click.pass_context
def create(ctx):
    # Create a new project
    # TODO
    ctx
