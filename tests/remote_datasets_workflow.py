from pypadre.app.padre_app import p_app

assert p_app.offline
p_app.authenticate("hmafnan", "test")
assert p_app.offline is False

list_of_datasets = p_app.local_backend.datasets.list()
print("Datasets in local before deleting")
print(list_of_datasets)
for ds in list_of_datasets:
    p_app.datasets.delete(ds)

print("Datasets in local")
print(p_app.local_backend.datasets.list())

print("Datasets on server")
print(p_app.remote_backend.datasets.list())

downloads = p_app.datasets.search_downloads("iris")
print("Datasets available on openMl for Iris")
print(downloads)
print("Downloaded External Datasets: ")
for externel_dataset in p_app.datasets.download(downloads):
    p_app.datasets.put(externel_dataset)

print("Total available datasets in Padre")
for ds in p_app.datasets.list():
    p_app.datasets.print(ds)