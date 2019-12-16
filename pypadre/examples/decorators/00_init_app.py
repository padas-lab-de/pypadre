import os

from pypadre.pod.app import PadreConfig
from pypadre.pod.app.padre_app import PadreAppFactory

config_path = os.path.join(os.path.expanduser("~"), ".padre-example.cfg")
workspace_path = os.path.join(os.path.expanduser("~"), ".pypadre-example")

config = PadreConfig(config_file=config_path)
config.set("backends", str([
    {
        "root_dir": workspace_path
    }
]))
app = PadreAppFactory.get(config)
