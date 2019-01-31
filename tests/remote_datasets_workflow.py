from padre.app.padre_app import pypadre

assert pypadre.offline
pypadre.config.authenticate("hmafnan", "test")
assert pypadre.offline == False

list_of_datasets = pypadre.local_backend.datasets.list()
for ds in list_of_datasets:
    pypadre.datasets.delete(ds)

print("Datasets in local")
print(pypadre.local_backend.datasets.list())

print("Datasets on server")
print(pypadre.remote_backend.datasets.list_dataset())

downloads = pypadre.datasets.search_downloads("Iris")
print("Datasets available on openMl for Iris")
print(downloads)
print("Downloaded External Datasets: ")
for externel_dataset in pypadre.datasets.download_external(downloads):
    pypadre.datasets.put(externel_dataset, upload=True)

print("After Downloading External, available datasets on local")
print(pypadre.local_backend.datasets.list())

print("After Downloading External On Remote")
print(pypadre.remote_backend.datasets.list_dataset())