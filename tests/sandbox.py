"""File for some testings"""
import padre.backend.file as parep
import padre.utils as pu
from padre.backend.http import PadreHTTPClient
from padre.ds_import import load_sklearn_toys

if __name__ == "__main__":
    client = PadreHTTPClient();
    r = client.do_get("datasets")
    print (r)

    for ds in load_sklearn_toys():
        print (ds)
        client.upload_dataset(ds, True)




if __name__ == '__mainx__':
    pass




    # import numpy as np
    # import pandas as pd
    # import io
    # import msgpack as mp
    # import pyarrow as pa
    # df2 = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)),
    #                    columns=['a', 'b', 'c', 'd', 'e'])
    # buffer = io.BytesIO()
    # df2.to_msgpack(buffer, append=False)
    # x = pa.serialize_pandas(df2)
    # y = pa.deserialize_pandas(x)
    # print(y)
    # view = buffer.getbuffer()
    # buffer.seek(0)
    # print(view)
    # x = mp.load(buffer)
    # print(x)





if __name__ == '__mainx__':
    rest_repository = parep.PadreRestClient()
    pu.print_dicts_as_table([d.metadata for d in rest_repository.list_datasets()],
                            heads=["name", "type", "description", "attributes"])
    getted = []
    for d in rest_repository.list_datasets():
        ds = rest_repository.get_dataset(d.id)
        getted.append(ds)

    pu.print_dicts_as_table([d.metadata for d in getted],
                                        heads=["name", "type", "description", "attributes"])

    #unittest.main()
