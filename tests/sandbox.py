"""File for some testings"""
from padre import ds_import

if __name__ == '__main__':
    from padre.app.padre_app import pypadre
    pypadre.offline = True # work only local
    # pypadre._http_repo.authenticate("test", "mgrani")
    print("Available datasets:")
    datasets = pypadre.datasets.list(prnt=True)
    for ds in datasets:
        pypadre.datasets.delete(ds, False)
    print("Available datasets after deletion:")
    pypadre.datasets.list(prnt=True)
    for ds in ds_import.load_sklearn_toys():
        pypadre.datasets.put(ds, upload=False)
    datasets = pypadre.datasets.list()
    for ds in datasets:
        dataset = pypadre.datasets.get(ds, force_download=False)
        pypadre.datasets.print_dataset_details(dataset)
    print(pypadre.offline)
