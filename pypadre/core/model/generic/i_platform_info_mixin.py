import platform
from abc import ABCMeta, abstractmethod

from pypadre.core.base import MetadataMixin


class PlatformInfoMixin(MetadataMixin):
    """ This is the abstract class for all entities holding platform metadata."""
    __metaclass__ = ABCMeta

    @classmethod
    def _tablefy_register_columns(cls):
        super()._tablefy_register_columns()
        cls.tablefy_register("platform_info")

    @abstractmethod
    def __init__(self, *args, metadata=None, **kwargs):

        sys_info = dict()
        sys_info["processor"] = platform.processor()
        sys_info["machine"] = platform.machine()
        sys_info["system"] = platform.system()
        sys_info["platform"] = platform.platform()
        sys_info["platform_version"] = platform.version()
        sys_info["node_name"] = platform.node()
        sys_info["python_version"] = platform.python_version()
        metadata["platform_info"] = sys_info

        super().__init__(metadata=metadata, **kwargs)

    @property
    def platform_info(self):
        return self.metadata["platform_info"]
