# This code has been created with the tensorflow documentation from
# "https://www.tensorflow.org/api_docs/python/tf/layers"
# All the layer names, types used for parameters and parameter names are defined below so that
# magic strings do not appear. This is to ensure that errors, if any are easy to be found and rectified.
# Each layer is defined with a name which identifies the layer.
# It is given in all upper case to ensure that even if the user changes the case, the code would work
# Each layer is defined as a dictionary with the key as the layer name.
# The dictionary contains the implementation path of the layer for dynamic loading of the module,
# and a dictionary of the parameters of that layer
# The parameter dictionary contains the name of the parameter, with the possible types in a list,
# whether the parameter is optional or not, and if it is optional the default value of the parameter
# The final dictionary to be dumped to JSON

from copy import deepcopy

# Datatypes and other keywords
optional = "optional"
_type = "type"
path = "path"
_int = "int"
_tuple = "tuple"
_bool = "bool"
_float = "float"
_list = "list"
_str = "str"
_tensor = "Tensor"
_iterable = "iterable"
default = "default"
params = "params"

# Parameters
pool_size = "pool_size"
stride = "stride"
padding = "padding"
data_format = "data_format"
name = "name"

# Layers
average_pooling1d = "AVGPOOL1D"
avgpool2d = "AVGPOOL2D"
avgpool3d = "AVGPOOL3D"

layers_dict = dict()

# Average Pooling 1D
pool_size_dict = dict()
pool_size_dict[_type] = [_int, _list]
pool_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _list]
stride_dict[optional] = False

padding_dict = dict()
padding_dict[_type] = [_str]
padding_dict[optional] = True
padding_dict[default] =  'valid'

data_format_dict = dict()
data_format_dict[_type] = [_str]
data_format_dict[optional] = True
data_format_dict[default] = 'channels_last'

name_dict = dict()
name_dict[_type] = [_str]
name_dict[optional] = True
name_dict[default] = None

average_pooling1d_params = dict()
average_pooling1d_params[pool_size] = pool_size_dict
average_pooling1d_params[stride] = stride_dict
average_pooling1d_params[padding] = padding_dict
average_pooling1d_params[data_format] = data_format_dict
average_pooling1d_params[name] = name_dict

average_pooling1d_dict = dict()
average_pooling1d_dict[path] = "tf.layers.average_pooling1d"
average_pooling1d_dict[params] = deepcopy(average_pooling1d_params)

layers_dict[average_pooling1d] = deepcopy(average_pooling1d_dict)