#!/usr/bin/env python

from google.protobuf.internal import encoder
import pandas as pd
import padre.protobuffer.protobuf.datasetV1_pb2 as proto
import numpy as np
import random
import time
import padre.datasets

def start_measure_time():
    return time.process_time()


def end_measure_time(t):
    elapsed_time = time.process_time() - t
    print("... %.3f sec" % elapsed_time)


def set_cell(pb_row, df_cell):
    pb_cell = pb_row.cells.add()

    df_cell_type = type(df_cell)

    # data type mapping: https://developers.google.com/protocol-buffers/docs/proto3#scalar
    if df_cell_type is int:
        pb_cell.int32_t = df_cell
    elif df_cell_type is float:
        pb_cell.float_t = df_cell
    elif df_cell_type is str:
        pb_cell.string_t = df_cell
    elif df_cell_type is bool:
        pb_cell.bool_t = df_cell


def write_delimited_pb_msg(binary, pb_msg):
    pb_msg_serialized = pb_msg.SerializeToString()
    length_varint = encoder._VarintBytes(len(pb_msg_serialized))
    binary.write(length_varint + pb_msg_serialized)


# df = pd.read_csv('../sample data/iris.data', header=None, names=["sepal length [cm]", "sepal width [cm]", "petal length [cm]", "petal width [cm]", "classification"])
# df = pd.DataFrame(np.random.randn(80000, 100)) #columns=list('ABCDEFGHIJ'))

# data = {'PBstringColumn': ['Some String value', 'Another String value'], 'PBint32Column': [12345, 67890], 'PBfloatColumn': [12345.67890, 98765.4321], 'PBboolColumn': [True, False]}
# df = pd.DataFrame(data=data)

# example column data type conversion 
# df[['col1', ]] = df[['col1', ]].astype(str)
# print(df.dtypes)

# print('---------------------------------------------------------')
# print('Dataframe (rows, columns): ' + str(df.shape))
# print('---------------------------------------------------------')
# print('Dataframe datatype per column:')
# print(df.dtypes)
# print('---------------------------------------------------------')
# print('Dataframe:')
# print(df)
# print('---------------------------------------------------------')
# print()

print("Building ProtoBuffer ...")
#t = start_measure_time()


"""
# build metadata; add header name values or use iteration if nothing is given
pb_dataframe_meta = proto.DataFrameMeta()
pb_dataframe_meta.headers[:] = [str(header) for header in list(df)]
write_delimited_pb_msg(binary, pb_dataframe_meta)

# add rows and cell values
for df_row in df.itertuples():
	pb_row = proto.DataFrameRow()

	for i, df_cell in enumerate(df_row):
		# avoid dataframe index column
		if i is 0:
			continue

		set_cell(pb_row, df_cell)

	write_delimited_pb_msg(binary, pb_row)

end_measure_time(t)
"""

def send_protobuffer(dataset):
    binary = open("protobinVTest", "wb")
    #dataset=padre.datasets.Dataset()
    dataframe=dataset.data
    data_attributes=dataset._binary._attributes
    dataframe_format=dataset._binary_format

    pb_dataframe_meta = proto.DataFrameMeta()
    pb_dataframe_meta.headers[:] = [str(header) for header in list(dataframe)]
    write_delimited_pb_msg(binary, pb_dataframe_meta)
    k=0
    #for df_row in dataframe.itertuples():
    l=list()
    print("start")
    for z in range(5000):
        pb_row = proto.DataFrameRow()
        #print(k)
        k=k+1
        #for i, df_cell in enumerate(df_row):
        for i  in range(200):
            # avoid dataframe index column
            if i is 0:
                continue
            set_cell(pb_row, 3)
        #l.append(pb_row)
        #write_delimited_pb_msg(binary, pb_row)
    binary.close()
    print("done")
"""
# build multiple rows and cells without a dataframe (less memory usage)
rows_count = 1000
columns_count = 100

# build metadata
pb_dataframe_meta = proto.DataFrameMeta()

# build and write metadata headers
for i in range(1, columns_count + 1):
    pb_dataframe_meta.headers.append('col' + str(i))

write_delimited_pb_msg(binary, pb_dataframe_meta)

# build and write rows and cell values
for i_r in range(0, rows_count):
    pb_row = proto.DataFrameRow()

    for i_c in range(0, columns_count):
        set_cell(pb_row, 'STRING')  # random integers: random.randrange(0, 1000)

    write_delimited_pb_msg(binary, pb_row)

end_measure_time(t)

binary.close()
"""

"""
# read dataset from binary file
print("Reading dataset from protobuffer binary ... ", end="")
start_measure_time()
f = open("binaries/dataset.protobinV1", "rb")
dataset = proto.DataFrameMeta()
# TODO: read delimited: dataset.ParseFromString(f.read())
f.close()
end_measure_time(t)
print(dataset)
print("---------------------------------------------------------")
"""
