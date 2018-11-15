#!/usr/bin/env python
import tempfile

from requests_toolbelt import MultipartEncoder
from google.protobuf.internal import encoder
from google.protobuf.internal import decoder
import pandas as pd
import padre.protobuffer.protobuf.datasetV1_pb2 as proto
import time
import requests
import os

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


def read_delimited_pb_msg(binary, msg_pos, pb_msg):
    """Used to decode the protobuffer message. Modifies pb_msg to contain the decoded information. Returns the value of
    the position of the binary, where to continue.

    Args:
        binary (binary): The data to be decoded.
        msg_pos (int): The start-index of where to start reading of the binary.
        pb_msg (proto.DataFrameRow()/proto.DataFrameMeta() Where the decoded binary is stored in.

    Returns:
        int The end-index of the the decoded line of the Protobuffer in the binary.

    """
    msg_len, parse_pos = decoder._DecodeVarint(binary, msg_pos)
    #print(msg_len)
    parse_pos_end = parse_pos+msg_len
    pb_msg_binary = binary[parse_pos:parse_pos_end]
    pb_msg.ParseFromString(pb_msg_binary)
    return parse_pos_end


def write_delimited_pb_msg(binary, pb_msg):
    """Used to encode the protobuffer message. Serializes the pb_msg and stores it in the binary.

    Args:
        binary (binary): The file, that the Protobuffer message should be stored to.
        pb_msg (proto.DataFrameRow()/proto.DataFrameMeta() The message that gets serialized to be stored in the binary.
    """
    pb_msg_serialized = pb_msg.SerializeToString()
    length_varint = encoder._VarintBytes(len(pb_msg_serialized))
    binary.write(length_varint + pb_msg_serialized)


def createProtobuffer(dataset,binary):
    #did=ident
    #path = path + "/datasets/temp/sendproto" + did + ".protobinV1"
    #os.makedirs(os.path.dirname(path), exist_ok=True)
    #binary = tempfile.TemporaryFile(mode='w+b')
    #binary = open(path, "wb")
    pd_dataframe = dataset.data


    pb_dataframe_meta = proto.Meta()
    pb_dataframe_meta.headers[:] = [str(header) for header in list(pd_dataframe)]

    # pb_dataframe_meta.headers[:] = ["jsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasdjsadlfjasldkfjalksdjfalsdflasdlfköaskdjflaksdjfklöasd"
    #                                for header in pd_dataframe.columns.values]

    # print(pb_dataframe_meta)
    write_delimited_pb_msg(binary, pb_dataframe_meta)
    t = start_measure_time()

    # add rows and cell values

    """
    #alternative writing to file in some cases faster, in some slower
    for df_row in pd_dataframe.itertuples():
        pb_row = proto.DataFrameRow()
        for i, df_cell in enumerate(df_row):
            # avoid dataframe index column
            if i is 0:
                continue

            set_cell(pb_row, df_cell)

        write_delimited_pb_msg(binary, pb_row)

    """

    col_list = []
    for col_name in pd_dataframe.columns.values.tolist():
        col_list.append(pd_dataframe[col_name])

    for row in zip(*col_list):
        pb_row = proto.DataRow()
        for entry in row:
            #print(type(entry))
            set_cell(pb_row, entry)
        write_delimited_pb_msg(binary, pb_row)


    proto_enlarged=False
    file_size=binary.tell()
    if(file_size<10000):
        proto_enlarged=True
        binary.seek(0)

        pb_dataframe_meta = proto.Meta()
        header_entries = list(pd_dataframe)
        header_entries.append("INVALID_COLUMN")
        pb_dataframe_meta.headers[:] = [str(header) for header in header_entries]

        # print(pb_dataframe_meta)
        write_delimited_pb_msg(binary, pb_dataframe_meta)
        t = start_measure_time()

        row_value = int((1200-(file_size/11)) / pd_dataframe.shape[0]) * "INVALID"
        col_list.append(pd.Series([row_value for i in range(pd_dataframe.shape[0])]))
        pb_row = proto.DataRow()
        for row in zip(*col_list):
            pb_row = proto.DataRow()
            for entry in row:
                # print(type(entry))
                set_cell(pb_row, entry)
            write_delimited_pb_msg(binary, pb_row)

    print("time taken for writing dataset to file:", end=" ")
    end_measure_time(t)
    # sending saved file

    return proto_enlarged


def send_Dataset(dataset,did,auth_token,binary,url="http://localhost:8080"):
    """Sends the dataset to the server in form of protobuffer.

    Args:
        dataset (padre.Dataset()): The Dataset whose content should be transferred to the server.
        did (str): The did of the dataset at the server, that should be filled with the protobuffer-messages.
        auth_token (str): The Token for identification.
        path (str): path of the pypadre directory
    """
    #path=path+ "/datasets/temp/sendproto"+ident+".protobinV1"
    #binary = open(path, "wb")


    hed = {'Authorization': 'Bearer ' + auth_token}
    url = url+"/api/datasets/" + str(did) + "/binaries"


    m=MultipartEncoder(fields={"field0": ("fname", binary,"application/x.padre.dataset.v1+protobuf")})
    hed["Content-Type"]=m.content_type
    try:
        r = requests.post(url, data=m, headers=hed)
    except requests.ConnectionError:
        r.close()
        requests.session().close()
        try:
            r = requests.post(url, data=m, headers=hed)
        except requests.ConnectionError:
            print("Unsuccessful upload of filedata of Dataset: "+did)
            r.close()
            requests.session().close()



    print(r.request.headers)
    print("responese:")
    print(r.content)
    r.close()
    requests.session().close()




def get_Server_Dataframe(did,auth_token,url="http://localhost:8080"):
    """Fetches the requested Dataset from the Server. Returns the Dataset as padre.Dataset().

    Args:
        did (str): id of the requested dataset
        auth_token (str): The Token for identification.

    Returns:
        padre.Dataset() A dataset containing with the requested data.
    """

    hed = {'Authorization': 'Bearer ' + auth_token}
    url = url+"/api/datasets/" + str(did)+"/binaries"
    response = requests.get(url, headers=hed)
    print(response.content)
    print(response)
    pb_data=response.content
    #pb_data=open("sendproto59.protobinV1","rb").read()
    response.close()
    requests.session().close()

    response=None
    t = start_measure_time()
    # read and build metadata
    pb_dataframe_meta = proto.Meta()
    pb_parse_pos = read_delimited_pb_msg(pb_data, 0, pb_dataframe_meta)

    # read and build row and cell values
    row_count = 0
    df_lines=[]
    while pb_parse_pos < len(pb_data):
        pb_dataframe_row = proto.DataRow()
        pb_parse_pos = read_delimited_pb_msg(pb_data, pb_parse_pos, pb_dataframe_row)
        #use patternmatching to get the data of the decoded protobuffer
        data=(str(pb_dataframe_row).split("cells {\n  ")[1:])
        df_line=[]
        for cell in data:
            cell=cell[0:-3]
            firstLetter=cell[0:1]
            if(firstLetter is "s"):
                df_line.append(cell[11:-1])
            elif(firstLetter is "f"):
                df_line.append(float(cell[9:]))
            elif(firstLetter is "i"):
                df_line.append(int(cell[9:]))
            else:
                if firstLetter[7:8] is 'T':
                    df_line.append(True)
                else:
                    df_line.append(False)
        df_lines.append(df_line)
        """
        if(row_count==0):
            df = pd.DataFrame([df_line])
        else:
            df.loc[row_count] = df_line

            """
            #df.append([df_line])
        row_count += 1

    df = pd.DataFrame(df_lines)

    print('Row count: ' + str(row_count),end=" ")
    print("time taken for building protobuffer")
    end_measure_time(t)

    return df