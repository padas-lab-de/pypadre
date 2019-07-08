import os
from ..constants import  RESOURCE_DIRECTORY_PATH
import ctypes


class ResourceDirectory:

    def create_directory(self):
        # TODO create a corresponding configuration object. look up best practices
        data_dir = os.path.expanduser(RESOURCE_DIRECTORY_PATH)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir


FILE_ATTRIBUTE_HIDDEN = 0x02

def write_hidden(file_name, data):
    """
    Cross platform hidden file writer.
    see https://stackoverflow.com/questions/25432139/python-cross-platform-hidden-file
    """
    # For *nix add a '.' prefix.
    prefix = '.' if os.name != 'nt' else ''
    file_name = prefix + file_name

    # Write file.
    with open(file_name, 'w') as f:
        f.write(data)

    # For windows set file attribute.
    if os.name == 'nt':
        ret = ctypes.windll.kernel32.SetFileAttributesW(file_name,
                                                        FILE_ATTRIBUTE_HIDDEN)
        if not ret: # There was an error.
            raise ctypes.WinError()

class _const:

    class ConstError(TypeError): pass

    def __setattr__(self,name,value):
        if self.__dict__.has_key(name):
            raise self.ConstError("Can't rebind const(%s)" % name)
        self.__dict__[name]=value
