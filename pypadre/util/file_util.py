import os
import re


def get_dir(root_dir, name):
    return get_path(root_dir, str(name))


def get_path(root_dir, name, create=True):
    # internal get or create path function
    _dir = os.path.expanduser(os.path.join(root_dir, name))
    if not os.path.exists(_dir) and create:
        os.mkdir(_dir)
    return _dir


def dir_list(root_dir, matcher, strip_postfix=""):
    files = [f for f in os.listdir(root_dir) if f.endswith(strip_postfix)]
    if matcher is not None:
        rid = re.compile(matcher)
        files = [f for f in files if rid.match(f)]

    if len(strip_postfix) == 0:
        return files
    else:
        return [file[:-1*len(strip_postfix)] for file in files
            if file is not None and len(file) >= len(strip_postfix)]