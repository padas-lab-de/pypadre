"""File for some testings"""
from pypadre import ds_import

if __name__ == '__main__':
    from pypadre.app.padre_app import p_app
    p_app.offline = True # work only local
    # pypadre._http_repo.authenticate("test", "mgrani")
    print("Available datasets:")
    datasets = p_app.datasets.list(prnt=True)
    for ds in datasets:
        p_app.datasets.delete(ds, False)
    print("Available datasets after deletion:")
    p_app.datasets.list(prnt=True)
    for ds in ds_import.load_sklearn_toys():
        p_app.datasets.put(ds, upload=False)
    datasets = p_app.datasets.list()
    for ds in datasets:
        dataset = p_app.datasets.get(ds, force_download=False)
        p_app.datasets.print(dataset)
    print(p_app.offline)
