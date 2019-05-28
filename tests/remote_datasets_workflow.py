from padre.app.padre_app import pypadre

assert pypadre.offline
pypadre.authenticate("hmafnan", "test")
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

downloads = pypadre.datasets.search_downloads("iris")
print("Datasets available on openMl for Iris")
print(downloads)
print("Downloaded External Datasets: ")
for externel_dataset in pypadre.datasets.download(downloads):
    pypadre.datasets.put(externel_dataset, upload=True)

print("Total available datasets in Padre")
for ds in pypadre.datasets.list():
    pypadre.datasets.print_dataset_details(ds)