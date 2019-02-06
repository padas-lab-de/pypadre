from padre.app.padre_app import pypadre

assert pypadre.offline
pypadre.config.authenticate("hmafnan", "test")
assert pypadre.offline == False

list_of_datasets = pypadre.local_backend.datasets.list()
print("Datasets in local before deleting")
print(list_of_datasets)
for ds in list_of_datasets:
    pypadre.datasets.delete(ds)

print("Datasets in local")
print(pypadre.local_backend.datasets.list())

print("Datasets on server")
print(pypadre.remote_backend.datasets.list())

downloads = pypadre.datasets.search_downloads("breast-cancer")
print("Datasets available on openMl for Breast cancer")
print(downloads)
print("Downloaded External Datasets: ")
for externel_dataset in pypadre.datasets.download(downloads):
    pypadre.datasets.put(externel_dataset, upload=True)

print("After Downloading External, available datasets on local and remote")
print(pypadre.datasets.list())