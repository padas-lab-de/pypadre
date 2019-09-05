import os


def get_path(root_dir, name, create=True):
    # internal get or create path function
    _dir = os.path.expanduser(os.path.join(root_dir, name))
    if not os.path.exists(_dir) and create:
        os.makedirs(_dir)
    return _dir