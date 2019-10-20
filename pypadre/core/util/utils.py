import json
import platform
from typing import Tuple


class _Const:

    class ConstError(TypeError): pass

    def __setattr__(self, name, value):
        if self.__dict__.has_key(name):
            raise self.ConstError("Can't rebind const(%s)" % name)
        self.__dict__[name]=value


def get_sys_info():
    # TODO: Implement the gathering of system information as dynamic code
    # TODO: Remove hard coded strings.
    # This function collects all system related info in a dictionary
    sys_info = dict()
    sys_info["processor"] = platform.processor()
    sys_info["machine"] = platform.machine()
    sys_info["system"] = platform.system()
    sys_info["platform"] = platform.platform()
    sys_info["platform_version"] = platform.version()
    sys_info["node_name"] = platform.node()
    sys_info["python_version"] = platform.python_version()
    return sys_info


def _merge_dict_class_vars(clz, clz_var:str, limit_clz):
    var = dict()
    for pclz in clz.mro():
        if issubclass(pclz, limit_clz) and hasattr(pclz, clz_var):
            var = {**var, **getattr(pclz, clz_var)}
    return var


def _merge_dict_class_vars(clz, clz_var:str, limit_clz):
    var = dict()
    for pclz in clz.mro():
        if issubclass(pclz, limit_clz) and hasattr(pclz, clz_var):
            var = {**var, **getattr(pclz, clz_var)}
    return var


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False


def remove_cached(cache, id):
    to_remove = []
    for k, v in cache.items():
        # Clear cache for item with id or if it is None because we don't know for which id the cache is.
        # TODO use the key to look for clearing
        if v is None or v.id == id:
            to_remove.append(k)

    for k in to_remove:
        cache.pop(k)


def unpack(kwargs_obj: dict, *args):
    """
    Unpacks a dict object into a tuple. You can pass tuples for setting default values.
    :param kwargs_obj:
    :param args:
    :return:
    """
    empty = object()
    arg_list = []
    for entry in args:
        if isinstance(entry, str):
            arg_list.append(kwargs_obj.get(entry))
        elif isinstance(entry, Tuple):
            key, *rest = entry
            default = empty if len(rest) == 0 else rest[0]
            if default is empty:
                arg_list.append(kwargs_obj.get(key))
            else:
                arg_list.append(kwargs_obj.get(key, default))
        else:
            raise ValueError("Pass a tuple or string not: " + str(entry))
    return tuple(arg_list)
