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
_dict = "dict"
_iterable = "iterable"
default = "default"
params = "params"

# Parameters
pool_size = "pool_size"
stride = "stride"
padding = "padding"
data_format = "data_format"
name = "name"
axis = "axis"
momentum = "momentum"
epsilon = "epsilon"
center = "center"
scale = "scale"
beta_initializer = "beta_intiializer"
gamma_initializer = "gamma_initializer"
moving_mean_initializer = "moving_mean_initializer"
moving_variance_initializer = "moving_variance_initializer"
beta_regularizer = "beta_regularizer"
gamma_regularizer = "gamma_regularizer"
beta_constraint = "beta_constraint"
gamma_constraint = "gamma_constraint"
renorm = "renorm"
renorm_clipping = "renorm_clipping"
renorm_momentum = "renorm_momentum"
fused = "fused"
trainable = "trainable"
virtual_batch_size = "virtual_batch_size"
adjustment = "adjustment"
filters = "filters"
kernel_size = "kernel_size"
strides = "strides"
dilation_rate = "dilation_rate"
activation = "activation"
use_bias = "use_bias"
kernel_initializer = "kernel_initializer"
bias_initializer = "bias_initializer"
kernel_regularizer = "kernel_regularizer"
bias_regularizer = "bias_regularizer"
activity_regularizer = "activity_regularizer"
kernel_constraint = "kernel_constraint"
bias_constraint = "bias_constrain"

# Layers
average_pooling1d = "AVGPOOL1D"
average_pooling2d = "AVGPOOL2D"
average_pooling3d = "AVGPOOL3D"
batch_norm = "BATCHNORM"
conv1d = "CONV1D"
conv2d = "CONV2D"
conv3d = "CONV3D"

def test_dictionary(completed_object_list, input_dict):
    """
    Test function for testing the dictionaries and reporting any inconsistencies found

    :param completed_object_list: Completed layers/functions/objects
    :param input_dict: The dictionary containing all the completed layers/functions/objects

    :return: None
    """

    layer_paths = []
    for layer_name in completed_object_list:

        if input_dict.get(layer_name, None) is None:
            print('Missing entry for ' + layer_name)
            continue

        curr_layer_path = input_dict.get(layer_name, None).get(path)

        if curr_layer_path is None:
            print('No path found for layer:' + layer_name)

        elif curr_layer_path in layer_paths:
            print('Path duplicate found for path:' + curr_layer_path)

        else:
            layer_paths.append(curr_layer_path)

        layer_params_dict = input_dict.get(layer_name).get(params, "NOPARAMSFOUND")

        if layer_params_dict is None:
            continue

        elif layer_params_dict == "NOPARAMSFOUND":
            print('No parameters found for layer:' + layer_name)
            continue

        for param_name in layer_params_dict:
            param_dict = layer_params_dict.get(param_name, None)
            if param_dict is None:
                continue

            if type(param_dict) is str:
                print(param_dict)

            default_value = param_dict.get(default, 'DEFAULTPARAMETERNOTFOUND')

            if param_dict.get(optional) is True and default_value == 'DEFAULTPARAMETERNOTFOUND':
                print("Default parameter missing for parameter " + param_name + " in layer " + layer_name)

            elif param_dict.get(optional) is False and param_dict.get(default, None) is not None:
                print("Default parameter given for compulsory parameter " + param_name + " in layer " + layer_name)

            possible_types = param_dict.get(_type, None)

            if possible_types is None:
                print('Types not specified for parameter ' + param_name + ' for layer ' + layer_name)

            if type(possible_types) is not list:
                print('Types wrongly specified for parameter ' + param_name + ' for layer ' + layer_name)

    set_diff = (set(input_dict.keys()).difference(set(completed_object_list)))
    if len(set_diff) > 0:
        print("Following keys are present in the dictionary but not in completed list")
        print(set_diff)

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
average_pooling1d_dict[path] = "tf.layers.AveragePooling1D"
average_pooling1d_dict[params] = deepcopy(average_pooling1d_params)

layers_dict[average_pooling1d] = deepcopy(average_pooling1d_dict)

# Average Pooling 2D
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

average_pooling2d_params = dict()
average_pooling2d_params[pool_size] = pool_size_dict
average_pooling2d_params[stride] = stride_dict
average_pooling2d_params[padding] = padding_dict
average_pooling2d_params[data_format] = data_format_dict
average_pooling2d_params[name] = name_dict

average_pooling2d_dict = dict()
average_pooling2d_dict[path] = "tf.layers.AveragePooling2D"
average_pooling2d_dict[params] = deepcopy(average_pooling2d_params)

layers_dict[average_pooling2d] = deepcopy(average_pooling2d_dict)

# Average Pooling 3D
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

average_pooling3d_params = dict()
average_pooling3d_params[pool_size] = pool_size_dict
average_pooling3d_params[stride] = stride_dict
average_pooling3d_params[padding] = padding_dict
average_pooling3d_params[data_format] = data_format_dict
average_pooling3d_params[name] = name_dict

average_pooling3d_dict = dict()
average_pooling3d_dict[path] = "tf.layers.AveragePooling3D"
average_pooling3d_dict[params] = deepcopy(average_pooling3d_params)

layers_dict[average_pooling3d] = deepcopy(average_pooling3d_dict)

# Batch Normalization
axis_dict = dict()
axis_dict[_type] = [_int]
axis_dict[optional] = True
axis_dict[default] = -1

momentum_dict = dict()
momentum_dict[_type] = [_float]
momentum_dict[optional] = True
momentum_dict[default] = 0.99

epsilon_dict = dict()
epsilon_dict[_type] = [_float]
epsilon_dict[optional] = True
epsilon_dict[default] = 0.001

center_dict = dict()
center_dict[_type] = [_bool]
center_dict[optional] = True
center_dict[default] = True

scale_dict = dict()
scale_dict[_type] = [_bool]
scale_dict[optional] = True
scale_dict[default] = True

# Currently this is set as string although it can take a function
beta_initializer_dict = dict()
beta_initializer_dict[_type] = [_str]
beta_initializer_dict[optional] = True
beta_initializer_dict[default] = "tf.zeros_initializer()"

# Currently this is set as string although it can take a function
gamma_initializer_dict = dict()
gamma_initializer_dict[_type] = [_str]
gamma_initializer_dict[optional] = True
gamma_initializer_dict[default] = "tf.ones_initializer()"

# Currently this is set as string although it can take a function
moving_mean_initializer_dict = dict()
moving_mean_initializer_dict[_type] = [_str]
moving_mean_initializer_dict[optional] = True
moving_mean_initializer_dict[default] = "tf.zeros_initializer()"

# Currently this is set as string although it can take a function
moving_variance_initializer_dict = dict()
moving_variance_initializer_dict[_type] = [_str]
moving_variance_initializer_dict[optional] = True
moving_variance_initializer_dict[default] = "tf.ones_initializer()"

# Currently this is set as string although it can take a function
beta_regularizer_dict = dict()
beta_regularizer_dict[_type] = [_str]
beta_regularizer_dict[optional] = True
beta_regularizer_dict[default] = None

# Currently this is set as string although it can take a function
gamma_regularizer_dict = dict()
gamma_regularizer_dict[_type] = [_str]
gamma_regularizer_dict[optional] = True
gamma_regularizer_dict[default] = None

# Currently this is set as string although it can take a function
beta_constraint_dict = dict()
beta_constraint_dict[_type] = [_str]
beta_constraint_dict[optional] = True
beta_constraint_dict[default] = None

# Currently this is set as string although it can take a function
gamma_constraint_dict = dict()
gamma_constraint_dict[_type] = [_str]
gamma_constraint_dict[optional] = True
gamma_constraint_dict[default] = None

renorm_dict = dict()
renorm_dict[_type] = [_bool]
renorm_dict[optional] = True
renorm_dict[default] = False

renorm_clipping_dict = dict()
renorm_clipping_dict[_type] = [_dict]
renorm_clipping_dict[optional] = True
renorm_clipping_dict[default] = None

renorm_momentum_dict = dict()
renorm_momentum_dict[_type] = [_float]
renorm_momentum_dict[optional] = True
renorm_momentum_dict[default] = 0.99

fused_dict = dict()
fused_dict[_type] = [_bool]
fused_dict[optional] = True
fused_dict[default] = None

trainable_dict = dict()
trainable_dict[_type] = [_bool]
trainable_dict[optional] = True
trainable_dict[default] = True

virtual_batch_size_dict = dict()
virtual_batch_size_dict[_type] = [_int]
virtual_batch_size_dict[optional] = True
virtual_batch_size_dict[default] = None

adjustment_dict = dict()
adjustment_dict[_type] = [_str]
adjustment_dict[optional] = True
adjustment_dict[default] = None

name_dict = dict()
name_dict[_type] = [_str]
name_dict[optional] = True
name_dict[default] = None

batch_norm_params = dict()
batch_norm_params[axis] = axis_dict
batch_norm_params[momentum] = momentum_dict
batch_norm_params[epsilon] = epsilon_dict
batch_norm_params[center] = center_dict
batch_norm_params[scale] = scale_dict
batch_norm_params[beta_initializer] = beta_initializer_dict
batch_norm_params[gamma_initializer] = gamma_initializer_dict
batch_norm_params[moving_mean_initializer] = moving_mean_initializer_dict
batch_norm_params[moving_variance_initializer] = moving_variance_initializer_dict
batch_norm_params[beta_regularizer] = beta_regularizer_dict
batch_norm_params[gamma_regularizer] = gamma_regularizer_dict
batch_norm_params[beta_constraint] = beta_constraint_dict
batch_norm_params[gamma_constraint] = gamma_constraint_dict
batch_norm_params[renorm] = renorm_dict
batch_norm_params[renorm_clipping] = renorm_clipping_dict
batch_norm_params[renorm_momentum] = renorm_momentum_dict
batch_norm_params[fused] = fused_dict
batch_norm_params[trainable] = trainable_dict
batch_norm_params[adjustment] = adjustment_dict
batch_norm_params[name] = name_dict

batch_norm_dict = dict()
batch_norm_dict[path] = "tf.layers.BatchNormalization"
batch_norm_dict[params] = deepcopy(batch_norm_params)

layers_dict[batch_norm] = deepcopy(batch_norm_dict)

# 1D Convolution
filters_dict = dict()
filters_dict[_type] = [_int]
filters_dict[optional] = False

kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple, _list]
kernel_size_dict[optional] = False

strides_dict = dict()
strides_dict[_type] = [_int, _tuple, _list]
strides_dict[optional] = True
strides_dict[default] = 1

padding_dict = dict()
padding_dict[_type] = [_str]
padding_dict[optional] = True
padding_dict[default] = 'valid'

data_format_dict = dict()
data_format_dict[_type] = [_str]
data_format_dict[optional] = True
data_format_dict[default] = 'channels_last'

dilation_rate_dict = dict()
dilation_rate_dict[_type] = [_int, _tuple, _list]
dilation_rate_dict[optional] = True
dilation_rate_dict[default] = 1

# Activation function, currently using str as parameter
activation_dict = dict()
activation_dict[_type] = [_str]
activation_dict[optional] = True
activation_dict[default] = None

use_bias_dict = dict()
use_bias_dict[_type] =  [_bool]
use_bias_dict[optional] = True
use_bias_dict[default] = True

# Currently string is given
kernel_initializer_dict = dict()
kernel_initializer_dict[_type] = [_str]
kernel_initializer_dict[optional] = True
kernel_initializer_dict[default] = None

# Can accept a function, currently a string is given as param
bias_initializer_dict = dict()
bias_initializer_dict[_type] = [_str]
bias_initializer_dict[optional] = True
bias_initializer_dict[default] = "tf.zeros_initializer"

# Currently param given as string
kernel_regularizer_dict = dict()
kernel_initializer_dict[_type] = [_str]
kernel_initializer_dict[optional] = True
kernel_initializer_dict[default] = None

# Currently param is given as a string
bias_regularizer_dict = dict()
bias_regularizer_dict[_type] = [_str]
bias_regularizer_dict[optional] = True
bias_regularizer_dict[default] = None

# Currently param is given as a string
activity_regularizer_dict = dict()
activity_regularizer_dict[_type] = [_str]
activity_regularizer_dict[optional] = True
activity_regularizer_dict[default] = None

# Currently param is given as a string
kernel_constraint_dict = dict()
kernel_constraint_dict[_type] = [_str]
kernel_constraint_dict[optional] =  True
kernel_constraint_dict[default] = None

# Currently param is given as a string
bias_constraint_dict = dict()
bias_constraint_dict[_type] = [_str]
bias_constraint_dict[optional] = True
bias_constraint_dict[default] = None

trainable_dict = dict()
trainable_dict[_type] = [_bool]
trainable_dict[optional] = True
trainable_dict[default] = True

virtual_batch_size_dict = dict()
virtual_batch_size_dict[_type] = [_int]
virtual_batch_size_dict[optional] = True
virtual_batch_size_dict[default] = None

name_dict = dict()
name_dict[_type] = [_str]
name_dict[optional] = True
name_dict[default] = None

conv1d_params = dict()
conv1d_params[filters] = filters_dict
conv1d_params[kernel_size] = kernel_size_dict
conv1d_params[strides] = strides_dict
conv1d_params[padding] = padding_dict
conv1d_params[data_format] = data_format_dict
conv1d_params[dilation_rate] = dilation_rate_dict
conv1d_params[activation] = activation_dict
conv1d_params[use_bias] = use_bias_dict
conv1d_params[kernel_initializer] = kernel_initializer_dict
conv1d_params[bias_initializer] = bias_initializer_dict
conv1d_params[kernel_regularizer] = kernel_regularizer_dict
conv1d_params[bias_regularizer] = bias_regularizer_dict
conv1d_params[activity_regularizer] = activity_regularizer_dict
conv1d_params[kernel_constraint] = kernel_constraint_dict
conv1d_params[bias_constraint] = bias_constraint_dict
conv1d_params[trainable] = trainable_dict
conv1d_params[name] = name_dict

conv1d_dict = dict()
conv1d_dict[path] = "tf.layers.Conv1D"
conv1d_dict[params] = deepcopy(conv1d_params)

layers_dict[conv1d] = deepcopy(conv1d_dict)
