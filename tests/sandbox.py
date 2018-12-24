"""File for some testings"""


if __name__ == '__main__':
    from padre.app.padre_app import pypadre
    pypadre._http_repo.authenticate("test", "mgrani")
    print(pypadre.offline)
