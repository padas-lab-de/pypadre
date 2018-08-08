
# This code has been created with the pytorch documentation from "https://pytorch.org/docs/stable/nn.html"
# Random samples parameter not included in FractionalMaxPool2D as the documentation does not specify the type
# All the layer names, types used for parameters and parameter names are defined below so that
# recurring magic strings do not appear. This is to ensure that errors, if any are easy to be found and rectified.
# Each layer is defined with a name which identifies the layer.
# It is given in all upper case to ensure that even if the user changes the case, the code would work
# Each layer is defined as a dictionary with the key as the layer name.
# The dictionary contains the implementation path of the layer for dynamic loading of the module,
# and a dictionary of the parameters of that layer
# The parameter dictionary contains the name of the parameter, with the possible types in a list,
# whether the parameter is optional or not, and if it is optional the default value of the parameter
# The final dictionary to be dumped to JSON

from copy import deepcopy
import json
import os

# The different strings present in the dictionaries are declared below
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

# The different layers defined are declared below
conv1d = "CONV1D"
conv2d = "CONV2D"
conv3d = "CONV3D"
transpose1d = "CONVTRANSPOSE1D"
transpose2d = "CONVTRANSPOSE2D"
transpose3d = "CONVTRANSPOSE3D"
unfold = "UNFOLD"
fold = "FOLD"

maxpool1d = "MAXPOOL1D"
maxpool2d = "MAXPOOL2D"
maxpool3d = "MAXPOOL3D"
maxunpool1d = "MAXUNPOOL1D"
maxunpool2d = "MAXUNPOOL2D"
maxunpool3d = "MAXUNPOOL3D"
avgpool1d = "AVGPOOL1D"
avgpool2d = "AVGPOOL2D"
avgpool3d = "AVGPOOL3D"
fractionalmaxpool2d = "FRACTIONALMAXPOOL2D"
lppool1d = "LPPOOL1D"
lppool2d = "LPPOOL2D"
adaptivemaxpool1d = "ADAPTIVEMAXPOOL1D"
adaptivemaxpool2d = "ADAPTIVEMAXPOOL2D"
adaptivemaxpool3d = "ADAPTIVEMAXPOOL3D"
adaptiveavgpool1d = "ADAPTIVEAVGPOOL1D"
adaptiveavgpool2d = "ADAPTIVEAVGPOOL2D"
adaptiveavgpool3d = "ADAPTIVEAVGPOOL3D"

reflectionpad1d = "REFLECTIONPAD1D"
reflectionpad2d = "REFLECTIONPAD2D"
replicationpad1d = "REPLICATIONPAD1D"
replicationpad2d = "REPLICATIONPAD2D"
replicationpad3d = "REPLICATIONPAD3D"
zeropad2d = "ZEROPAD2D"
constantpad1d = "CONSTANTPAD1D"
constantpad2d = "CONSTANTPAD2D"
constantpad3d = "CONSTANTPAD3D"

elu = "ELU"
hardshrink = "HARDSHRINK"
hardtanh = "HARDTANH"
leakyrelu = "LEAKYRELU"
logsigmoid = "LOGSIGMOID"
prelu = "PRELU"
relu = "RELU"
relu6 = "RELU6"
rrelu = "RRELU"
selu = "SELU"
sigmoid = "SIGMOID"
softplus = "SOFTPLUS"
softshrink = "SOFTSHRINK"
softsign = "SOFTSIGN"
tanh = "TANH"
tanhshrink = "TANHSHRINK"
threshold = "THRESHOLD"
softmin = "SOFTMIN"
softmax = "SOFTMAX"
softmax2d = "SOFTMAX2D"
logsoftmax = "LOGSOFTMAX"
adaptivelogsoftmaxwithloss = "ADAPTIVELOGSOFTMAXWITHLOSS"

batchnorm1d = "BATCHNORM1D"
batchnorm2d = "BATCHNORM2D"
batchnorm3d = "BATCHNORM3D"
groupnorm = "GROUPNORM"
instancenorm1d = "INSTANCENORM1D"
instancenorm2d = "INSTANCENORM2D"
instancenorm3d = "INSTANCENORM3D"
layernorm = "LAYERNORM"
localresponsenorm = "LOCALRESPONSENORM"

linear = "LINEAR"
bilinear = "BILINEAR"

dropout = "DROPOUT"
dropout2d = "DROPOUT2D"
dropout3d = "DROPOUT3D"
alphadropout = "ALPHADROPOUT"

# The different parameters for the layers are declared below
in_channels = "in_channels"
out_channels = "out_channels"
kernel_size = "kernel_size"
stride = "stride"
padding = "padding"
dilation = "dilation"
groups = "groups"
bias = "bias"
return_indices = "return_indices"
output_size = "output_size"
output_ratio = "output_ratio"
ceil_mode = "ceil_mode"
count_include = "count_include"
norm_type = "norm_type"
value = "value"
alpha = "alpha"
inplace = "inplace"
lambd = "lambd"
min_val = "min_val"
max_val = "max_val"
min_value = "min_value"
max_value = "max_value"
negative_slope = "negative_slope"
num_parameters = "num_parameters"
init = "init"
lower = "lower"
upper = "upper"
beta = "beta"
param_threshold = "threshold"
dim = "dim"
in_features = "in_features"
n_classes = "n_classes"
cutoffs = "cutoffs"
div_value = "div_value"
head_bias = "head_bias"
num_features = "num_features"
eps = "eps"
momentum = "momentum"
affine = "affine"
track_running_status = "track_running_status"
num_groups = "num_groups"
num_channels = "num_channels"
normalized_shape = "normalized_shape"
elementwise_affine = "elementwise_affine"
size = "size"
k = "k"
out_features = "out_features"
in1_features = "in1_features"
in2_features = "in2_features"
p = "p"

# This part is for the testing of the entered layers
# Tests are
# 1. All layers should have unique paths
# 2. Default parameters should have a value associated with it
# 3. Compulsory parameters should not have a value along with it
# 4. The types possible should be a list
# 5. Verify that all defined layers are present within the layers_dict
# 6. Verify that all the layers in layers_dict is present within the completed layers list

# Create a list with all the layer names in it


def test_dictionary(completed_object_list, input_dict):
    layer_paths = []
    for layer_name in completed_object_list:

        if input_dict.get(layer_name, None) is None:
            print('Missing entry for ' + layer_name)
            continue

        curr_layer_path = input_dict.get(layer_name, None).get(path)

        if curr_layer_path in layer_paths:
            print('Path duplicate found for path:' + curr_layer_path)

        else:
            layer_paths.append(curr_layer_path)

        layer_params_dict = input_dict.get(layer_name).get(params, None)

        if layer_params_dict is None:
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

# Convolution 1D Layer Definition
in_channels_dict = dict()
in_channels_dict[_type] = [_int]
in_channels_dict[optional] = False

out_channels_dict = dict()
out_channels_dict[_type] = [_int]
out_channels_dict[optional] = False

kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = 1

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

dilation_dict = dict()
dilation_dict[_type] = [_int, _tuple]
dilation_dict[optional] = True
dilation_dict[default] = 1

groups_dict = dict()
groups_dict[_type] = [_int]
groups_dict[optional] = True
groups_dict[default] = 1

bias_dict = dict()
bias_dict[_type] = [_bool]
bias_dict[optional] = True
bias_dict[default] = True

conv_params = dict()
conv_params[in_channels] = deepcopy(in_channels_dict)
conv_params[out_channels] = deepcopy(out_channels_dict)
conv_params[kernel_size] = deepcopy(kernel_size_dict)
conv_params[stride] = deepcopy(stride_dict)
conv_params[padding] = deepcopy(padding_dict)
conv_params[dilation] = deepcopy(dilation_dict)
conv_params[groups] = deepcopy(groups_dict)
conv_params[bias] = deepcopy(bias_dict)

conv_dict = dict()
conv_dict[path] = "torch.nn.Conv1d"
conv_dict[params] = deepcopy(conv_params)

layers_dict[conv1d] = deepcopy(conv_dict)

# Convolution 2D Layer Definition
in_channels_dict = dict()
in_channels_dict[_type] = [_int]
in_channels_dict[optional] = False

out_channels_dict = dict()
out_channels_dict[_type] = [_int]
out_channels_dict[optional] = False

kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = 1

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

dilation_dict = dict()
dilation_dict[_type] = [_int, _tuple]
dilation_dict[optional] = True
dilation_dict[default] = 1

groups_dict = dict()
groups_dict[_type] = [_int]
groups_dict[optional] = True
groups_dict[default] = 1

bias_dict = dict()
bias_dict[_type] = [_bool]
bias_dict[optional] = True
bias_dict[default] = True

conv_params = dict()
conv_params[in_channels] = deepcopy(in_channels_dict)
conv_params[out_channels] = deepcopy(out_channels_dict)
conv_params[kernel_size] = deepcopy(kernel_size_dict)
conv_params[stride] = deepcopy(stride_dict)
conv_params[padding] = deepcopy(padding_dict)
conv_params[dilation] = deepcopy(dilation_dict)
conv_params[groups] = deepcopy(groups_dict)
conv_params[bias] = deepcopy(bias_dict)

conv_dict = dict()
conv_dict[path] = "torch.nn.Conv2d"
conv_dict[params] = deepcopy(conv_params)

layers_dict[conv2d] = deepcopy(conv_dict)

# Convolution 3D Layer Definition
in_channels_dict = dict()
in_channels_dict[_type] = [_int]
in_channels_dict[optional] = False

out_channels_dict = dict()
out_channels_dict[_type] = [_int]
out_channels_dict[optional] = False

kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = 1

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

dilation_dict = dict()
dilation_dict[_type] = [_int, _tuple]
dilation_dict[optional] = True
dilation_dict[default] = 1

groups_dict = dict()
groups_dict[_type] = [_int]
groups_dict[optional] = True
groups_dict[default] = 1

bias_dict = dict()
bias_dict[_type] = [_bool]
bias_dict[optional] = True
bias_dict[default] = True

conv_params = dict()
conv_params[in_channels] = deepcopy(in_channels_dict)
conv_params[out_channels] = deepcopy(out_channels_dict)
conv_params[kernel_size] = deepcopy(kernel_size_dict)
conv_params[stride] = deepcopy(stride_dict)
conv_params[padding] = deepcopy(padding_dict)
conv_params[dilation] = deepcopy(dilation_dict)
conv_params[groups] = deepcopy(groups_dict)
conv_params[bias] = deepcopy(bias_dict)

conv_dict = dict()
conv_dict[path] = "torch.nn.Conv3d"
conv_dict[params] = deepcopy(conv_params)

layers_dict[conv3d] = deepcopy(conv_dict)

# Transposed 1D Convolution Layer
in_channels_dict = dict()
in_channels_dict[_type] = [_int]
in_channels_dict[optional] = False

out_channels_dict = dict()
out_channels_dict[_type] = [_int]
out_channels_dict[optional] = False

kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = 1

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

output_padding_dict = dict()
output_padding_dict[_type] = [_int, _tuple]
output_padding_dict[optional] = True
output_padding_dict[default] = 0

groups_dict = dict()
groups_dict[_type] = [_int]
groups_dict[optional] = True
groups_dict[default] = 1

bias_dict = dict()
bias_dict[_type] = [_bool]
bias_dict[optional] = True
bias_dict[default] = True

dilation_dict = dict()
dilation_dict[_type] = [_int, _tuple]
dilation_dict[optional] = True
dilation_dict[default] = 1


transpose_params = dict()
transpose_params[in_channels] = deepcopy(in_channels_dict)
transpose_params[out_channels] = deepcopy(out_channels_dict)
transpose_params[kernel_size] = deepcopy(kernel_size_dict)
transpose_params[stride] = deepcopy(stride_dict)
transpose_params[padding] = deepcopy(padding_dict)
transpose_params[dilation] = deepcopy(dilation_dict)
transpose_params[groups] = deepcopy(groups_dict)
transpose_params[bias] = deepcopy(bias_dict)

transpose_dict = dict()
transpose_dict[path] = "torch.nn.ConvTranspose1d"
transpose_dict[params] = deepcopy(transpose_params)

layers_dict[transpose1d] = deepcopy(transpose_dict)

# Transposed 2D Convolution Layer
in_channels_dict = dict()
in_channels_dict[_type] = [_int]
in_channels_dict[optional] = False

out_channels_dict = dict()
out_channels_dict[_type] = [_int]
out_channels_dict[optional] = False

kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = 1

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

output_padding_dict = dict()
output_padding_dict[_type] = [_int, _tuple]
output_padding_dict[optional] = True
output_padding_dict[default] = 0

groups_dict = dict()
groups_dict[_type] = [_int]
groups_dict[optional] = True
groups_dict[default] = 1

bias_dict = dict()
bias_dict[_type] = [_bool]
bias_dict[optional] = True
bias_dict[default] = True

dilation_dict = dict()
dilation_dict[_type] = [_int, _tuple]
dilation_dict[optional] = True
dilation_dict[default] = 1


transpose_params = dict()
transpose_params[in_channels] = deepcopy(in_channels_dict)
transpose_params[out_channels] = deepcopy(out_channels_dict)
transpose_params[kernel_size] = deepcopy(kernel_size_dict)
transpose_params[stride] = deepcopy(stride_dict)
transpose_params[padding] = deepcopy(padding_dict)
transpose_params[dilation] = deepcopy(dilation_dict)
transpose_params[groups] = deepcopy(groups_dict)
transpose_params[bias] = deepcopy(bias_dict)

transpose_dict = dict()
transpose_dict[path] = "torch.nn.ConvTranspose2d"
transpose_dict[params] = deepcopy(transpose_params)

layers_dict[transpose2d] = deepcopy(transpose_dict)

# Transposed 3D Convolution Layer
in_channels_dict = dict()
in_channels_dict[_type] = [_int]
in_channels_dict[optional] = False

out_channels_dict = dict()
out_channels_dict[_type] = [_int]
out_channels_dict[optional] = False

kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = 1

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

output_padding_dict = dict()
output_padding_dict[_type] = [_int, _tuple]
output_padding_dict[optional] = True
output_padding_dict[default] = 0

groups_dict = dict()
groups_dict[_type] = [_int]
groups_dict[optional] = True
groups_dict[default] = 1

bias_dict = dict()
bias_dict[_type] = [_bool]
bias_dict[optional] = True
bias_dict[default] = True

dilation_dict = dict()
dilation_dict[_type] = [_int, _tuple]
dilation_dict[optional] = True
dilation_dict[default] = 1


transpose_params = dict()
transpose_params[in_channels] = deepcopy(in_channels_dict)
transpose_params[out_channels] = deepcopy(out_channels_dict)
transpose_params[kernel_size] = deepcopy(kernel_size_dict)
transpose_params[stride] = deepcopy(stride_dict)
transpose_params[padding] = deepcopy(padding_dict)
transpose_params[dilation] = deepcopy(dilation_dict)
transpose_params[groups] = deepcopy(groups_dict)
transpose_params[bias] = deepcopy(bias_dict)

transpose_dict = dict()
transpose_dict[path] = "torch.nn.ConvTranspose3d"
transpose_dict[params] = deepcopy(transpose_params)

layers_dict[transpose3d] = deepcopy(transpose_dict)

# Unfold Layer
kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = 1

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

dilation_dict = dict()
dilation_dict[_type] = [_int, _tuple]
dilation_dict[optional] = True
dilation_dict[default] = 1

unfold_params = dict()
unfold_params[kernel_size] = kernel_size_dict
unfold_params[stride] = stride_dict
unfold_params[padding] = padding_dict
unfold_params[dilation] = dilation_dict

unfold_dict = dict()
unfold_dict[path] = "torch.nn.Unfold"
unfold_dict[params] = deepcopy(unfold_params)

layers_dict[unfold] = deepcopy(unfold_dict)

# Fold Layer
output_size_dict = dict()
output_size_dict[_type] = [_int, _tuple]
output_size_dict[optional] = False

kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = 1

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

dilation_dict = dict()
dilation_dict[_type] = [_int, _tuple]
dilation_dict[optional] = True
dilation_dict[default] = 1

fold_params = dict()
fold_params[output_size] = output_size_dict
fold_params[kernel_size] = kernel_size_dict
fold_params[stride] = stride_dict
fold_params[padding] = padding_dict
fold_params[dilation] = dilation_dict

fold_dict = dict()
fold_dict[path] = "torch.nn.Fold"
fold_dict[params] = deepcopy(fold_params)

layers_dict[fold] = deepcopy(fold_dict)

# Max Pool 1D
kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = None 

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

dilation_dict = dict()
dilation_dict[_type] = [_int, _tuple]
dilation_dict[optional] = True
dilation_dict[default] = 1

return_indices_dict = dict()
return_indices_dict[_type] = [_bool]
return_indices_dict[optional] = True
return_indices_dict[default] = False

ceil_mode_dict = dict()
ceil_mode_dict[_type] = [_bool]
ceil_mode_dict[optional] = True
ceil_mode_dict[default] = False

maxpool1D_params = dict()
maxpool1D_params[kernel_size] = kernel_size_dict
maxpool1D_params[stride] = stride_dict
maxpool1D_params[padding] = padding_dict
maxpool1D_params[dilation] = dilation_dict
maxpool1D_params[return_indices] = return_indices_dict
maxpool1D_params[ceil_mode] = ceil_mode_dict

maxpool1D_dict = dict()
maxpool1D_dict[path] = "torch.nn.MaxPool1d"
maxpool1D_dict[params] = deepcopy(maxpool1D_params)

layers_dict[maxpool1d] = deepcopy(maxpool1D_dict)

# Max Pool 2D
kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = None 

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

dilation_dict = dict()
dilation_dict[_type] = [_int, _tuple]
dilation_dict[optional] = True
dilation_dict[default] = 1

return_indices_dict = dict()
return_indices_dict[_type] = [_bool]
return_indices_dict[optional] = True
return_indices_dict[default] = False

ceil_mode_dict = dict()
ceil_mode_dict[_type] = [_bool]
ceil_mode_dict[optional] = True
ceil_mode_dict[default] = False

maxpool2D_params = dict()
maxpool2D_params[kernel_size] = kernel_size_dict
maxpool2D_params[stride] = stride_dict
maxpool2D_params[padding] = padding_dict
maxpool2D_params[dilation] = dilation_dict
maxpool2D_params[return_indices] = return_indices_dict
maxpool2D_params[ceil_mode] = ceil_mode_dict

maxpool2D_dict = dict()
maxpool2D_dict[path] = "torch.nn.MaxPool2d"
maxpool2D_dict[params] = deepcopy(maxpool2D_params)

layers_dict[maxpool2d] = deepcopy(maxpool2D_dict)

# Max Pool 3D
kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = None 

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

dilation_dict = dict()
dilation_dict[_type] = [_int, _tuple]
dilation_dict[optional] = True
dilation_dict[default] = 1

return_indices_dict = dict()
return_indices_dict[_type] = [_bool]
return_indices_dict[optional] = True
return_indices_dict[default] = False

ceil_mode_dict = dict()
ceil_mode_dict[_type] = [_bool]
ceil_mode_dict[optional] = True
ceil_mode_dict[default] = False

maxpool3D_params = dict()
maxpool3D_params[kernel_size] = kernel_size_dict
maxpool3D_params[stride] = stride_dict
maxpool3D_params[padding] = padding_dict
maxpool3D_params[dilation] = dilation_dict
maxpool3D_params[return_indices] = return_indices_dict
maxpool3D_params[ceil_mode] = ceil_mode_dict

maxpool3D_dict = dict()
maxpool3D_dict[path] = "torch.nn.MaxPool3d"
maxpool3D_dict[params] = deepcopy(maxpool3D_params)

layers_dict[maxpool3d] = deepcopy(maxpool3D_dict)

# Max Unpool 1D
kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = None

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

maxunpool1d_params = dict()
maxunpool1d_params[kernel_size] = kernel_size_dict
maxunpool1d_params[stride] = stride_dict
maxunpool1d_params[padding] = padding_dict

maxunpool1d_dict = dict()
maxunpool1d_dict[path] = "torch.nn.MaxUnpool1d"
maxunpool1d_dict[params] = deepcopy(maxunpool1d_params)

layers_dict[maxunpool1d] = deepcopy(maxunpool1d_dict)

# Max Unpool 2D
kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = None

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

maxunpool2d_params = dict()
maxunpool2d_params[kernel_size] = kernel_size_dict
maxunpool2d_params[stride] = stride_dict
maxunpool2d_params[padding] = padding_dict

maxunpool2d_dict = dict()
maxunpool2d_dict[path] = "torch.nn.MaxUnpool2d"
maxunpool2d_dict[params] = deepcopy(maxunpool2d_params)

layers_dict[maxunpool2d] = deepcopy(maxunpool2d_dict)

# Max Unpool 3D
kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = None

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

maxunpool3d_params = dict()
maxunpool3d_params[kernel_size] = kernel_size_dict
maxunpool3d_params[stride] = stride_dict
maxunpool3d_params[padding] = padding_dict

maxunpool3d_dict = dict()
maxunpool3d_dict[path] = "torch.nn.MaxUnpool3d"
maxunpool3d_dict[params] = deepcopy(maxunpool3d_params)

layers_dict[maxunpool3d] = deepcopy(maxunpool3d_dict)

# Average pool 1D
kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = None

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

ceil_mode_dict = dict()
ceil_mode_dict[_type] = [_bool]
ceil_mode_dict[optional] = True
ceil_mode_dict[default] = False

count_include_dict = dict()
count_include_dict[_type] = [_bool]
count_include_dict[optional] = True
count_include_dict[default] = True

avgpool1d_params = dict()
avgpool1d_params[kernel_size] = kernel_size_dict
avgpool1d_params[stride] = stride_dict
avgpool1d_params[padding] = padding_dict
avgpool1d_params[ceil_mode] = ceil_mode_dict
avgpool1d_params[count_include] = count_include_dict

avgpool1d_dict = dict()
avgpool1d_dict[path] = "torch.nn.AvgPool1d"
avgpool1d_dict[params] = deepcopy(avgpool1d_params)

layers_dict[avgpool1d] = deepcopy(avgpool1d_dict)

# Average pool 2D
kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = None

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

ceil_mode_dict = dict()
ceil_mode_dict[_type] = [_bool]
ceil_mode_dict[optional] = True
ceil_mode_dict[default] = False

count_include_dict = dict()
count_include_dict[_type] = [_bool]
count_include_dict[optional] = True
count_include_dict[default] = True

avgpool2d_params = dict()
avgpool2d_params[kernel_size] = kernel_size_dict
avgpool2d_params[stride] = stride_dict
avgpool2d_params[padding] = padding_dict
avgpool1d_params[ceil_mode] = ceil_mode_dict
avgpool1d_params[count_include] = count_include_dict

avgpool2d_dict = dict()
avgpool2d_dict[path] = "torch.nn.AvgPool2d"
avgpool2d_dict[params] = deepcopy(avgpool2d_params)

layers_dict[avgpool2d] = deepcopy(avgpool2d_dict)

# Average pool 3D
kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = None

padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = True
padding_dict[default] = 0

ceil_mode_dict = dict()
ceil_mode_dict[_type] = [_bool]
ceil_mode_dict[optional] = True
ceil_mode_dict[default] = False

count_include_dict = dict()
count_include_dict[_type] = [_bool]
count_include_dict[optional] = True
count_include_dict[default] = True

avgpool3d_params = dict()
avgpool3d_params[kernel_size] = kernel_size_dict
avgpool3d_params[stride] = stride_dict
avgpool3d_params[padding] = padding_dict
avgpool3d_params[ceil_mode] = ceil_mode_dict
avgpool3d_params[count_include] = count_include_dict

avgpool3d_dict = dict()
avgpool3d_dict[path] = "torch.nn.AvgPool3d"
avgpool3d_dict[params] = deepcopy(avgpool3d_params)

layers_dict[avgpool3d] = deepcopy(avgpool3d_dict)

# Fractional Max Pool 2D
kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

output_size_dict = dict()
output_size_dict[_type] = [_int, _tuple]
output_size_dict[optional] = True
output_size_dict[default] = None

output_ratio_dict = dict()
output_ratio_dict[_type] = [_float]
output_ratio_dict[optional] = None

return_indices_dict = dict()
return_indices_dict[_type] = [_bool]
return_indices_dict[optional] = True
return_indices_dict[default] = False

fractionalmaxpool2d_params = dict()
fractionalmaxpool2d_params[kernel_size] = kernel_size_dict
fractionalmaxpool2d_params[output_size] = output_size_dict
fractionalmaxpool2d_params[output_ratio] = output_ratio_dict
fractionalmaxpool2d_params[return_indices] = return_indices_dict

fractionalmaxpool2d_dict = dict()
fractionalmaxpool2d_dict[path] = "torch.nn.FractionalMaxPool2d"
fractionalmaxpool2d_dict[params] = deepcopy(fractionalmaxpool2d_params)

layers_dict[fractionalmaxpool2d] = deepcopy(fractionalmaxpool2d_dict)

# LP Pool 1D
norm_type_dict = dict()
norm_type_dict[_type] = [_int]
norm_type_dict[optional] = False

kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = None

ceil_mode_dict = dict()
ceil_mode_dict[_type] = [_bool]
ceil_mode_dict[optional] = True
ceil_mode_dict[default] = False

lppool1d_params = dict()
lppool1d_params[norm_type] = norm_type_dict
lppool1d_params[kernel_size] = kernel_size_dict
lppool1d_params[stride] = stride_dict
lppool1d_params[ceil_mode] = ceil_mode_dict

lppool1d_dict = dict()
lppool1d_dict[path] = "torch.nn.LPPool1d"
lppool1d_dict[params] = deepcopy(lppool1d_params)

layers_dict[lppool1d] = deepcopy(lppool1d_dict)

# LP Pool 2D
norm_type_dict = dict()
norm_type_dict[_type] = [_int]
norm_type_dict[optional] = False

kernel_size_dict = dict()
kernel_size_dict[_type] = [_int, _tuple]
kernel_size_dict[optional] = False

stride_dict = dict()
stride_dict[_type] = [_int, _tuple]
stride_dict[optional] = True
stride_dict[default] = None

ceil_mode_dict = dict()
ceil_mode_dict[_type] = [_bool]
ceil_mode_dict[optional] = True
ceil_mode_dict[default] = False

lppool2d_params = dict()
lppool2d_params[norm_type] = norm_type_dict
lppool2d_params[kernel_size] = kernel_size_dict
lppool2d_params[stride] = stride_dict
lppool2d_params[ceil_mode] = ceil_mode_dict

lppool2d_dict = dict()
lppool2d_dict[path] = "torch.nn.LPPool2d"
lppool2d_dict[params] = deepcopy(lppool2d_params)

layers_dict[lppool2d] = deepcopy(lppool2d_dict)

# Adaptive Max Pooling 1D
output_size_dict = dict()
output_size_dict[_type] = [_int, _tuple]
output_size_dict[optional] = False

return_indices_dict = dict()
return_indices_dict[_type] = [_bool]
return_indices_dict[optional] = True
return_indices_dict[default] = False

adaptivemaxpool1d_params = dict()
adaptivemaxpool1d_params[output_size] = output_size_dict
adaptivemaxpool1d_params[return_indices] = return_indices_dict

adaptivemaxpool1d_dict = dict()
adaptivemaxpool1d_dict[path] = "torch.nn.AdaptiveMaxPool1d"
adaptivemaxpool1d_dict[params] = deepcopy(adaptivemaxpool1d_params)

layers_dict[adaptivemaxpool1d] = deepcopy(adaptivemaxpool1d_dict)

# Adaptive Max Pooling 2D
output_size_dict = dict()
output_size_dict[_type] = [_int, _tuple]
output_size_dict[optional] = False

return_indices_dict = dict()
return_indices_dict[_type] = [_bool]
return_indices_dict[optional] = True
return_indices_dict[default] = False

adaptivemaxpool2d_params = dict()
adaptivemaxpool2d_params[output_size] = output_size_dict
adaptivemaxpool2d_params[return_indices] = return_indices_dict

adaptivemaxpool2d_dict = dict()
adaptivemaxpool2d_dict[path] = "torch.nn.AdaptiveMaxPool2d"
adaptivemaxpool2d_dict[params] = deepcopy(adaptivemaxpool2d_params)

layers_dict[adaptivemaxpool2d] = deepcopy(adaptivemaxpool2d_dict)

# Adaptive Max Pooling 3D
output_size_dict = dict()
output_size_dict[_type] = [_int, _tuple]
output_size_dict[optional] = False

return_indices_dict = dict()
return_indices_dict[_type] = [_bool]
return_indices_dict[optional] = True
return_indices_dict[default] = False

adaptivemaxpool3d_params = dict()
adaptivemaxpool3d_params[output_size] = output_size_dict
adaptivemaxpool3d_params[return_indices] = return_indices_dict

adaptivemaxpool3d_dict = dict()
adaptivemaxpool3d_dict[path] = "torch.nn.AdaptiveMaxPoo31d"
adaptivemaxpool3d_dict[params] = deepcopy(adaptivemaxpool3d_params)

layers_dict[adaptivemaxpool3d] = deepcopy(adaptivemaxpool3d_dict)

# Adaptive Average Pooling 1D
output_size_dict = dict()
output_size_dict[_type] = [_int, _tuple]
output_size_dict[optional] = False

adaptiveavgpool1d_params = dict()
adaptiveavgpool1d_params[output_size] = output_size_dict

adaptiveavgpool1d_dict = dict()
adaptiveavgpool1d_dict[path] = "torch.nn.AdaptiveAvgPool1d"
adaptiveavgpool1d_dict[params] = deepcopy(adaptiveavgpool1d_params)

layers_dict[adaptiveavgpool1d] = deepcopy(adaptiveavgpool1d_dict)

# Adaptive Average Pooling 2D
output_size_dict = dict()
output_size_dict[_type] = [_int, _tuple]
output_size_dict[optional] = False

adaptiveavgpool2d_params = dict()
adaptiveavgpool2d_params[output_size] = output_size_dict

adaptiveavgpool2d_dict = dict()
adaptiveavgpool2d_dict[path] = "torch.nn.AdaptiveAvgPool2d"
adaptiveavgpool2d_dict[params] = deepcopy(adaptiveavgpool2d_params)

layers_dict[adaptiveavgpool2d] = deepcopy(adaptiveavgpool2d_dict)

# Adaptive Average Pooling 3D
output_size_dict = dict()
output_size_dict[_type] = [_int, _tuple]
output_size_dict[optional] = False

adaptiveavgpool3d_params = dict()
adaptiveavgpool3d_params[output_size] = output_size_dict

adaptiveavgpool3d_dict = dict()
adaptiveavgpool3d_dict[path] = "torch.nn.AdaptiveAvgPool3d"
adaptiveavgpool3d_dict[params] = deepcopy(adaptiveavgpool3d_params)

layers_dict[adaptiveavgpool3d] = deepcopy(adaptiveavgpool3d_dict)

# Reflection Padding 1D
padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = False

reflectionpad1d_params = dict()
reflectionpad1d_params[padding] = padding_dict

reflectionpad1d_dict = dict()
reflectionpad1d_dict[path] = "torch.nn.ReflectionPad1d"
reflectionpad1d_dict[params] = deepcopy(reflectionpad1d_params)

layers_dict[reflectionpad1d] = deepcopy(reflectionpad1d_dict)

# Reflection Padding 2D
padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = False

reflectionpad2d_params = dict()
reflectionpad2d_params[padding] = padding_dict

reflectionpad2d_dict = dict()
reflectionpad2d_dict[path] = "torch.nn.ReflectionPad2d"
reflectionpad2d_dict[params] = deepcopy(reflectionpad2d_params)

layers_dict[reflectionpad2d] = deepcopy(reflectionpad2d_dict)

# Replication Padding 1D
padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = False

replicationpad1d_params = dict()
replicationpad1d_params[padding] = padding_dict

replicationpad1d_dict = dict()
replicationpad1d_dict[path] = "torch.nn.ReplicationPad1d"
replicationpad1d_dict[params] = deepcopy(replicationpad1d_params)

layers_dict[replicationpad1d] = deepcopy(replicationpad1d_dict)

# Replication Padding 2D
padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = False

replicationpad2d_params = dict()
replicationpad2d_params[padding] = padding_dict

replicationpad2d_dict = dict()
replicationpad2d_dict[path] = "torch.nn.ReplicationPad2d"
replicationpad2d_dict[params] = deepcopy(replicationpad2d_params)

layers_dict[replicationpad2d] = deepcopy(replicationpad2d_dict)

# Replication Padding 3D
padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = False

replicationpad3d_params = dict()
replicationpad3d_params[padding] = padding_dict

replicationpad3d_dict = dict()
replicationpad3d_dict[path] = "torch.nn.ReplicationPad3d"
replicationpad3d_dict[params] = deepcopy(replicationpad3d_params)

layers_dict[replicationpad3d] = deepcopy(replicationpad3d_dict)

# Zero Padding 2D
padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = False

zeropad2d_params = dict()
zeropad2d_params[padding] = padding_dict

zeropad2d_dict = dict()
zeropad2d_dict[path] = "torch.nn.ZeroPad2d"
zeropad2d_dict[params] = deepcopy(zeropad2d_params)

layers_dict[zeropad2d] = deepcopy(zeropad2d_dict)

# Constant Padding 1D
padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = False

value_dict = dict()
value_dict[_type] = [_int, _tuple]
value_dict[optional] = False

constantpad1d_params = dict()
constantpad1d_params[padding] = padding_dict
constantpad1d_params[value] = value_dict

constantpad1d_dict = dict()
constantpad1d_dict[path] = "torch.nn.ConstantPad1d"
constantpad1d_dict[params] = deepcopy(constantpad1d_params)

layers_dict[constantpad1d] = deepcopy(constantpad1d_dict)

# Constant Padding 2D
padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = False

value_dict = dict()
value_dict[_type] = [_int, _tuple]
value_dict[optional] = False

constantpad2d_params = dict()
constantpad2d_params[padding] = padding_dict
constantpad2d_params[value] = value_dict

constantpad2d_dict = dict()
constantpad2d_dict[path] = "torch.nn.ConstantPad2d"
constantpad2d_dict[params] = deepcopy(constantpad2d_params)

layers_dict[constantpad2d] = deepcopy(constantpad2d_dict)

# Constant Padding 3D
padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = False

value_dict = dict()
value_dict[_type] = [_int, _tuple]
value_dict[optional] = False

constantpad3d_params = dict()
constantpad3d_params[padding] = padding_dict
constantpad3d_params[value] = value_dict

constantpad3d_dict = dict()
constantpad3d_dict[path] = "torch.nn.ConstantPad3d"
constantpad3d_dict[params] = deepcopy(constantpad3d_params)

layers_dict[constantpad3d] = deepcopy(constantpad3d_dict)

# ELU
alpha_dict = dict()
alpha_dict[_type] = [_float]
alpha_dict[optional] = True
alpha_dict[default] = 1

inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

elu_params = dict()
elu_params[alpha] = alpha_dict
elu_params[inplace] = inplace_dict

elu_dict = dict()
elu_dict[path] = "torch.nn.ELU"
elu_dict[params] = deepcopy(elu_params)

layers_dict[elu] = deepcopy(elu_dict)

# Hardshrink
lambd_dict = dict()
lambd_dict[_type] = [_float]
lambd_dict[optional] = True
lambd_dict[default] = 0.5

hardshrink_params = dict()
hardshrink_params[lambd] = lambd_dict

hardshrink_dict = dict()
hardshrink_dict[path] = "torch.nn.Hardshrink"
hardshrink_dict[params] = deepcopy(hardshrink_params)

layers_dict[hardshrink] = deepcopy(hardshrink_dict)

# Hardtanh
min_val_dict = dict()
min_val_dict[_type] = [_float]
min_val_dict[optional] = True
min_val_dict[default] = -1

max_val_dict = dict()
max_val_dict[_type] = [_float]
max_val_dict[optional] = True
max_val_dict[default] = 1

inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

min_value_dict = dict()
min_value_dict[_type] = [_float]
min_value_dict[optional] = True
min_value_dict[default] = None

max_value_dict = dict()
max_value_dict[_type] = [_float]
max_value_dict[optional] = True
max_value_dict[default] = None

hardtanh_params = dict()
hardtanh_params[min_val] = min_val_dict
hardtanh_params[max_val] = max_val_dict
hardtanh_params[inplace] = inplace_dict
hardtanh_params[min_value] = min_value_dict
hardtanh_params[max_value] = max_value_dict

hardtanh_dict = dict()
hardtanh_dict[path] = "torch.nn.Hardtanh"
hardtanh_dict[params] = deepcopy(hardtanh_params)

layers_dict[hardtanh] = deepcopy(hardtanh_dict)

# Leaky ReLU
negative_slope_dict = dict()
negative_slope_dict[_type] = [_float]
negative_slope_dict[optional] = True
negative_slope_dict[default] = 0.01

inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

leakyrelu_params = dict()
leakyrelu_params[negative_slope] = negative_slope_dict
leakyrelu_params[inplace] = inplace_dict

leakyrelu_dict = dict()
leakyrelu_dict[path] = "torch.nn.LeakyReLU"
leakyrelu_dict[params] = deepcopy(leakyrelu_params)

layers_dict[leakyrelu] = deepcopy(leakyrelu_dict)

# Log Sigmoid
logsigmoid_dict = dict()
logsigmoid_dict[path] = "torch.nn.LogSigmoid"
logsigmoid_dict[params] = None

layers_dict[logsigmoid] = deepcopy(logsigmoid_dict)

# PReLU
num_parameters_dict = dict()
num_parameters_dict[_type] = [_int]
num_parameters_dict[optional] = True
num_parameters_dict[default] = 1

init_dict = dict()
init_dict[_type] = [_float]
init_dict[optional] = True
init_dict[default] = 0.25

prelu_params = dict()
prelu_params[num_parameters] = num_parameters_dict
prelu_params[init] = init_dict

prelu_dict = dict()
prelu_dict[path] = "torch.nn.PReLU"
prelu_dict[params] = deepcopy(prelu_params)

layers_dict[prelu] = deepcopy(prelu_dict)

# ReLU
inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

relu_params = dict()
relu_params[inplace] = inplace_dict

relu_dict = dict()
relu_dict[path] = "torch.nn.ReLU"
relu_dict[params] = deepcopy(relu_params)

layers_dict[relu] = deepcopy(relu_dict)

# ReLU6
inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

relu6_params = dict()
relu6_params[inplace] = inplace_dict

relu6_dict = dict()
relu6_dict[path] = "torch.nn.ReLU6"
relu6_dict[params] = deepcopy(relu6_params)

layers_dict[relu6] = deepcopy(relu6_dict)

# RReLU
lower_dict = dict()
lower_dict[_type] = [_float]
lower_dict[optional] = True
lower_dict[default] = 0.125

upper_dict = dict()
upper_dict[_type] = [_float]
upper_dict[optional] = True
upper_dict[default] = 0.3333333333333333333333

inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

rrelu_params = dict()
rrelu_params[lower] = lower_dict
rrelu_params[upper] = upper_dict
rrelu_params[inplace] = inplace_dict

rrelu_dict = dict()
rrelu_dict[path] = "torch.nn.RReLU"
rrelu_dict[params] = deepcopy(rrelu_params)

layers_dict[rrelu] = deepcopy(rrelu_dict)

# SELU
inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

selu_params = dict()
selu_params[inplace] = inplace_dict

selu_dict = dict()
selu_dict[path] = "torch.nn.SELU"
selu_dict[params] = deepcopy(selu_params)

layers_dict[selu] = deepcopy(selu_dict)

# Sigmoid
sigmoid_dict = dict()
sigmoid_dict[path] = "torch.nn.Sigmoid"
sigmoid_dict[params] = None

layers_dict[sigmoid] = deepcopy(sigmoid_dict)

# Softplus
beta_dict = dict()
beta_dict[_type] = [_float]
beta_dict[optional] = True
beta_dict[default] = 1

threshold_dict = dict()
threshold_dict[_type] = [_float]
threshold_dict[optional] = True
threshold_dict[default] = 20

softplus_params = dict()
softplus_params[beta] = beta_dict
softplus_params[param_threshold] = threshold_dict

softplus_dict = dict()
softplus_dict[path] = "torch.nn.Softplus"
softplus_dict[params] = deepcopy(softplus_params)

layers_dict[softplus] = deepcopy(softplus_dict)

# Softshrink
lambd_dict = dict()
lambd_dict[_type] = [_float]
lambd_dict[optional] = True
lambd_dict[default] = 0.5

softshrink_params = dict()
softshrink_params[lambd] = lambd_dict

softshrink_dict = dict()
softshrink_dict[path] = "torch.nn.Softshrink"
softshrink_dict[params] = deepcopy(softshrink_params)

layers_dict[softshrink] = deepcopy(softshrink_dict)

# Softsign
softsign_dict = dict()
softplus_dict[path] = "torch.nn.Softsign"
softplus_dict[params] = None

layers_dict[softsign] = deepcopy(softsign_dict)

# Tanh
tanh_dict = dict()
tanh_dict[path] = "torch.nn.Tanh"
tanh_dict[params] = None

layers_dict[tanh] = deepcopy(tanh_dict)

# Tanhshrink
tanhshrink_dict = dict()
tanhshrink_dict[path] = "torch.nn.Tanhshrink"
tanhshrink_dict[params] = None

layers_dict[tanhshrink] = deepcopy(tanhshrink_dict)

# Threshold
threshold_dict = dict()
threshold_dict[_type] = [_float]
threshold_dict[optional] = False

value_dict = dict()
value_dict[_type] = [_int, _tuple]
value_dict[optional] = False

inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

threshold_params_dict = dict()
threshold_params_dict[param_threshold] = threshold_dict
threshold_params_dict[value] = value_dict
threshold_params_dict[inplace] = inplace_dict

threshold_dict = dict()
threshold_dict[path] = "torch.nn.Threshold"
threshold_dict[params] = deepcopy(threshold_params_dict)

layers_dict[threshold] = deepcopy(threshold_dict)

# Softmin
dim_dict = dict()
dim_dict[_type] = [_int]
dim_dict[optional] = True
dim_dict[default] = None

softmin_params = dict()
softmin_params[dim] = dim_dict

softmin_dict = dict()
softmin_dict[path] = "torch.nn.Softmax"
softmin_dict[params] = deepcopy(softmin_params)

layers_dict[softmin] = deepcopy(softmin_dict)

# Softmax
dim_dict = dict()
dim_dict[_type] = [_int]
dim_dict[optional] = True
dim_dict[default] = None

softmax_params = dict()
softmax_params[dim] = dim_dict

softmax_dict = dict()
softmax_dict[path] = "torch.nn.Softmin"
softmax_dict[params] = deepcopy(softmax_params)

layers_dict[softmax] = deepcopy(softmax_dict)

# Softmax 2D
softmax2d_dict = dict()
softmax2d_dict[path] = "torch.nn.Softmax2d"
softmax2d_dict[params] = None

layers_dict[softmax2d] = deepcopy(softmax2d_dict)

# Log Softmax
dim_dict = dict()
dim_dict[_type] = [_int]
dim_dict[optional] = True
dim_dict[default] = None

logsoftmax_params = dict()
logsoftmax_params[dim] = dim_dict

logsoftmax_dict = dict()
logsigmoid_dict[path] = "torch.nn.LogSoftmax"
logsigmoid_dict[params] = deepcopy(logsoftmax_params)

layers_dict[logsoftmax] = deepcopy(logsigmoid_dict)

# Adaptive Log Softmax With Loss
in_features_dict = dict()
in_features_dict[_type] = [_int]
in_features_dict[optional] = False

n_classes_dict = dict()
n_classes_dict[_type] = [_int]
n_classes_dict[optional] = False

cutoffs_dict = dict()
cutoffs_dict[_type] = [_list]
cutoffs_dict[optional] = False

div_value_dict = dict()
div_value_dict[_type] = [_float]
div_value_dict[optional] = True
div_value_dict[default] = 4.0

head_bias_dict = dict()
head_bias_dict[_type] = [_bool]
head_bias_dict[optional] = True
head_bias_dict[default] = False

adaptivelogsoftmaxwithloss_params = dict()
adaptivelogsoftmaxwithloss_params[in_features] = in_features_dict
adaptivelogsoftmaxwithloss_params[n_classes] = n_classes_dict
adaptivelogsoftmaxwithloss_params[cutoffs] = cutoffs_dict
adaptivelogsoftmaxwithloss_params[div_value] = div_value_dict
adaptivelogsoftmaxwithloss_params[head_bias] = head_bias_dict

adaptivelogsoftmaxwithloss_dict = dict()
adaptivelogsoftmaxwithloss_dict[path] = "torch.nn.AdaaptiveLogSoftmaxWithLoss"
adaptivelogsoftmaxwithloss_dict[params] = deepcopy(adaptivelogsoftmaxwithloss_params)

layers_dict[adaptivelogsoftmaxwithloss] = deepcopy(adaptivelogsoftmaxwithloss_dict)

# Batch norm 1D
num_features_dict = dict()
num_features_dict[_type] = [_int]
num_features_dict[optional] = False

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-05

momentum_dict = dict()
momentum_dict[_type] = [_float]
momentum_dict[optional] = True
momentum_dict[default] = 0.1

affine_dict = dict()
affine_dict[_type] = [_bool]
affine_dict[optional] = True
affine_dict[default] = True

track_running_status_dict = dict()
track_running_status_dict[_type] = [_bool]
track_running_status_dict[optional] = True
track_running_status_dict[default] = True

batchnorm1d_params = dict()
batchnorm1d_params[num_features] = num_features_dict
batchnorm1d_params[eps] = eps_dict
batchnorm1d_params[momentum] = momentum_dict
batchnorm1d_params[affine] = affine_dict
batchnorm1d_params[track_running_status] = track_running_status_dict

batchnorm1d_dict = dict()
batchnorm1d_dict[path] = "torch.nn.BatchNorm1d"
batchnorm1d_dict[params] = deepcopy(batchnorm1d_params)

layers_dict[batchnorm1d] = deepcopy(batchnorm1d_dict)

# Batch norm 2D
num_features_dict = dict()
num_features_dict[_type] = [_int]
num_features_dict[optional] = False

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-05

momentum_dict = dict()
momentum_dict[_type] = [_float]
momentum_dict[optional] = True
momentum_dict[default] = 0.1

affine_dict = dict()
affine_dict[_type] = [_bool]
affine_dict[optional] = True
affine_dict[default] = True

track_running_status_dict = dict()
track_running_status_dict[_type] = [_bool]
track_running_status_dict[optional] = True
track_running_status_dict[default] = True

batchnorm2d_params = dict()
batchnorm2d_params[num_features] = num_features_dict
batchnorm2d_params[eps] = eps_dict
batchnorm2d_params[momentum] = momentum_dict
batchnorm2d_params[affine] = affine_dict
batchnorm2d_params[track_running_status] = track_running_status_dict

batchnorm2d_dict = dict()
batchnorm2d_dict[path] = "torch.nn.BatchNorm2d"
batchnorm2d_dict[params] = deepcopy(batchnorm2d_params)

layers_dict[batchnorm2d] = deepcopy(batchnorm2d_dict)

# Batch norm 3D
num_features_dict = dict()
num_features_dict[_type] = [_int]
num_features_dict[optional] = False

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-05

momentum_dict = dict()
momentum_dict[_type] = [_float]
momentum_dict[optional] = True
momentum_dict[default] = 0.1

affine_dict = dict()
affine_dict[_type] = [_bool]
affine_dict[optional] = True
affine_dict[default] = True

track_running_status_dict = dict()
track_running_status_dict[_type] = [_bool]
track_running_status_dict[optional] = True
track_running_status_dict[default] = True

batchnorm3d_params = dict()
batchnorm3d_params[num_features] = num_features_dict
batchnorm3d_params[eps] = eps_dict
batchnorm3d_params[momentum] = momentum_dict
batchnorm3d_params[affine] = affine_dict
batchnorm3d_params[track_running_status] = track_running_status_dict

batchnorm3d_dict = dict()
batchnorm3d_dict[path] = "torch.nn.BatchNorm3d"
batchnorm3d_dict[params] = deepcopy(batchnorm3d_params)

layers_dict[batchnorm3d] = deepcopy(batchnorm3d_dict)

# Group Norm
num_groups_dict = dict()
num_groups_dict[_type] = [_int]
num_groups_dict[optional] = False

num_channels_dict = dict()
num_channels_dict[_type] = [_int]
num_channels_dict[optional] = False

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-05

affine_dict = dict()
affine_dict[_type] = [_bool]
affine_dict[optional] = True
affine_dict[default] = True

groupnorm_params = dict()
groupnorm_params[num_groups] = num_groups_dict
groupnorm_params[num_channels] = num_channels_dict
groupnorm_params[eps] = eps_dict
groupnorm_params[affine] = affine_dict

groupnorm_dict = dict()
groupnorm_dict[path] = "torch.nn.GroupNorm"
groupnorm_dict[params] = deepcopy(groupnorm_params)

layers_dict[groupnorm] = deepcopy(groupnorm_dict)

# Instance norm 1D
num_features_dict = dict()
num_features_dict[_type] = [_int]
num_features_dict[optional] = False

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-05

momentum_dict = dict()
momentum_dict[_type] = [_float]
momentum_dict[optional] = True
momentum_dict[default] = 0.1

affine_dict = dict()
affine_dict[_type] = [_bool]
affine_dict[optional] = True
affine_dict[default] = True

track_running_status_dict = dict()
track_running_status_dict[_type] = [_bool]
track_running_status_dict[optional] = True
track_running_status_dict[default] = True

instancenorm1d_params = dict()
instancenorm1d_params[num_features] = num_features_dict
instancenorm1d_params[eps] = eps_dict
instancenorm1d_params[momentum] = momentum_dict
instancenorm1d_params[affine] = affine_dict
instancenorm1d_params[track_running_status] = track_running_status_dict

instancehnorm1d_dict = dict()
instancehnorm1d_dict[path] = "torch.nn.InstanceNorm1d"
instancehnorm1d_dict[params] = deepcopy(instancenorm1d_params)

layers_dict[instancenorm1d] = deepcopy(instancehnorm1d_dict)

# Instance norm 2D
num_features_dict = dict()
num_features_dict[_type] = [_int]
num_features_dict[optional] = False

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-05

momentum_dict = dict()
momentum_dict[_type] = [_float]
momentum_dict[optional] = True
momentum_dict[default] = 0.1

affine_dict = dict()
affine_dict[_type] = [_bool]
affine_dict[optional] = True
affine_dict[default] = True

track_running_status_dict = dict()
track_running_status_dict[_type] = [_bool]
track_running_status_dict[optional] = True
track_running_status_dict[default] = True

instancenorm2d_params = dict()
instancenorm2d_params[num_features] = num_features_dict
instancenorm2d_params[eps] = eps_dict
instancenorm2d_params[momentum] = momentum_dict
instancenorm2d_params[affine] = affine_dict
instancenorm2d_params[track_running_status] = track_running_status_dict

instancenorm2d_dict = dict()
instancenorm2d_dict[path] = "torch.nn.InstanceNorm2d"
instancenorm2d_dict[params] = deepcopy(instancenorm2d_params)

layers_dict[instancenorm2d] = deepcopy(instancenorm2d_dict)

# Instance norm 3D
num_features_dict = dict()
num_features_dict[_type] = [_int]
num_features_dict[optional] = False

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-05

momentum_dict = dict()
momentum_dict[_type] = [_float]
momentum_dict[optional] = True
momentum_dict[default] = 0.1

affine_dict = dict()
affine_dict[_type] = [_bool]
affine_dict[optional] = True
affine_dict[default] = True

track_running_status_dict = dict()
track_running_status_dict[_type] = [_bool]
track_running_status_dict[optional] = True
track_running_status_dict[default] = True

instancenorm3d_params = dict()
instancenorm3d_params[num_features] = num_features_dict
instancenorm3d_params[eps] = eps_dict
instancenorm3d_params[momentum] = momentum_dict
instancenorm3d_params[affine] = affine_dict
instancenorm3d_params[track_running_status] = track_running_status_dict

instancenorm3d_dict = dict()
instancenorm3d_dict[path] = "torch.nn.InstanceNorm3d"
instancenorm3d_dict[params] = deepcopy(instancenorm3d_params)

layers_dict[instancenorm3d] = deepcopy(instancenorm3d_dict)

# Layer Norm
normalized_shape_dict = dict()
normalized_shape_dict[_type] = [_int, _list]
normalized_shape_dict[optional] = False

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-05

elementwise_affine_dict = dict()
elementwise_affine_dict[_type] = [_bool]
elementwise_affine_dict[optional] = True
elementwise_affine_dict[default] = True

layernorm_params = dict()
layernorm_params[normalized_shape] = normalized_shape_dict
layernorm_params[eps] = eps_dict
layernorm_params[elementwise_affine] = elementwise_affine_dict

layernorm_dict = dict()
layernorm_dict[path] = "torch.nn.LayerNorm"
layernorm_dict[params] = deepcopy(layernorm_params)

layers_dict[layernorm] = deepcopy(layernorm_dict)

# Local Response Norm
size_dict = dict()
size_dict[_type] = [_int]
size_dict[optional] = False

alpha_dict = dict()
alpha_dict[_type] = [_float]
alpha_dict[optional] = True
alpha_dict[default] = 0.0001

beta_dict = dict()
beta_dict[_type] = [_float]
beta_dict[optional] = True
beta_dict[default] = 0.75

k_dict = dict()
k_dict[_type] = [_int]
k_dict[optional] = True
k_dict[default] = 1

localresponsenorm_params = dict()
localresponsenorm_params[size] = size_dict
localresponsenorm_params[alpha] = alpha_dict
localresponsenorm_params[beta] = beta_dict
localresponsenorm_params[k] = k_dict

localresponsenorm_dict = dict()
localresponsenorm_dict[path] = "torch.nn.LocalResponseNorm"
localresponsenorm_dict[params] = deepcopy(localresponsenorm_params)

layers_dict[localresponsenorm] = deepcopy(localresponsenorm_dict)

# Linear
in_features_dict = dict()
in_features_dict[_type] = [_int]
in_features_dict[optional] = False

out_features_dict = dict()
out_features_dict[_type] = [_int]
out_features_dict[optional] = False

bias_dict = dict()
bias_dict[_type] = [_bool]
bias_dict[optional] = True
bias_dict[default] = True

linear_params_dict = dict()
linear_params_dict[in_features] = in_features_dict
linear_params_dict[out_features] = out_features_dict
linear_params_dict[bias] = bias_dict

linear_dict = dict()
linear_dict[path] = "torch.nn.Linear"
linear_dict[params] = deepcopy(linear_params_dict)

layers_dict[linear] = deepcopy(linear_dict)

# Bilinear
in1_features_dict = dict()
in1_features_dict[_type] = [_int]
in1_features_dict[optional] = False

in2_features_dict = dict()
in2_features_dict[_type] = [_int]
in2_features_dict[optional] = False

out_features_dict = dict()
out_features_dict[_type] = [_int]
out_features_dict[optional] = False

bias_dict = dict()
bias_dict[_type] = [_bool]
bias_dict[optional] = True
bias_dict[default] = True

bilinear_params_dict = dict()
bilinear_params_dict[in1_features] = in1_features_dict
bilinear_params_dict[in2_features] = in2_features_dict
bilinear_params_dict[out_features] = out_features_dict
bilinear_params_dict[bias] = bias_dict

bilinear_dict = dict()
bilinear_dict[path] = "torch.nn.Bilinear"
bilinear_dict[params] = deepcopy(linear_params_dict)

layers_dict[bilinear] = deepcopy(bilinear_dict)

# Dropout
p_dict = dict()
p_dict[_type] = [_int]
p_dict[optional] = True
p_dict[default] = 0.5

inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

dropout_params = dict()
dropout_params[p] = p_dict
dropout_params[inplace] = inplace_dict

dropout_dict = dict()
dropout_dict[path] = "torch.nn.Dropout"
dropout_dict[params] = deepcopy(dropout_params)

layers_dict[dropout] = deepcopy(dropout_dict)

# Dropout 2D
p_dict = dict()
p_dict[_type] = [_int]
p_dict[optional] = True
p_dict[default] = 0.5

inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

dropout2d_params = dict()
dropout2d_params[p] = p_dict
dropout2d_params[inplace] = inplace_dict

dropout2d_dict = dict()
dropout2d_dict[path] = "torch.nn.Dropout2d"
dropout2d_dict[params] = deepcopy(dropout2d_params)

layers_dict[dropout2d] = deepcopy(dropout2d_dict)

# Dropout 3D
p_dict = dict()
p_dict[_type] = [_int]
p_dict[optional] = True
p_dict[default] = 0.5

inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

dropout3d_params = dict()
dropout3d_params[p] = p_dict
dropout3d_params[inplace] = inplace_dict

dropout3d_dict = dict()
dropout3d_dict[path] = "torch.nn.Dropout3d"
dropout3d_dict[params] = deepcopy(dropout3d_params)

layers_dict[dropout3d] = deepcopy(dropout3d_dict)

# Alpha Dropout
p_dict = dict()
p_dict[_type] = [_int]
p_dict[optional] = True
p_dict[default] = 0.5

inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

alphadropout_params = dict()
alphadropout_params[p] = p_dict
alphadropout_params[inplace] = inplace_dict

alphadropout_dict = dict()
alphadropout_dict[path] = "torch.nn.AlphaDropout"
alphadropout_dict[params] = deepcopy(alphadropout_params)

layers_dict[alphadropout] = deepcopy(alphadropout_dict)

completed_layers = [
                    conv1d, conv2d, conv3d, transpose1d, transpose2d, transpose3d, unfold, fold, maxpool1d, maxpool2d,
                    maxpool3d, maxunpool1d, maxunpool2d, maxunpool3d, avgpool1d, avgpool2d, avgpool3d,
                    fractionalmaxpool2d, lppool1d, lppool2d, adaptivemaxpool1d, adaptivemaxpool2d, adaptivemaxpool3d,
                    adaptiveavgpool1d, reflectionpad1d, reflectionpad2d, replicationpad1d, replicationpad2d,
                    replicationpad3d, zeropad2d, constantpad1d, constantpad2d, constantpad3d, adaptiveavgpool2d,
                    adaptiveavgpool3d,  elu, hardshrink, hardtanh, leakyrelu, logsigmoid, prelu, relu, relu6, rrelu,
                    selu, sigmoid, softplus, softshrink, softsign, tanh, tanhshrink, threshold, softmin, softmax,
                    softmax2d, logsoftmax, adaptivelogsoftmaxwithloss, batchnorm1d, batchnorm2d, batchnorm3d, groupnorm,
                    instancenorm1d, instancenorm2d, instancenorm3d, layernorm, localresponsenorm, linear, bilinear,
                    dropout, dropout2d, dropout3d, alphadropout
                    ]

test_dictionary(completed_object_list=completed_layers, input_dict=layers_dict)
# This part deals with the different types of transformations possible for torch framework
# The parameters are based on the documentation available at the link below
# https://pytorch.org/docs/stable/torchvision/transforms.html

transforms_dict = dict()
# Transform names
centercrop = 'CENTERCROP'
colorjitter = "COLORJITTER"
fivecrop = "FIVECROP"
grayscale = "GRAYSCALE"
lineartransformation = "LINEARTRANSFORMATION"
pad = "PAD"
randomaffine = "RANDOMAFFINE"
randomapply = "RANDOMAPPLY"
randomchoise = "RANDOMCHOICE"
randomcrop = "RANDOMCROP"
randomgrayscale = "RANDOMGRAYSCALE"
randomhorizontalflip = "RANDOMHORIZONTALFLIP"
randomorder = "RANDOMORDER"
randomresizedcrop = "RANDOMRESIZEDCROP"
randomrotation = "RANDOMROTATION"
randomsizedcrop = "RANDOMSIZEDCROP"
randomverticalflip = "RANDOMVERTICALFLIP"
resize = "RESIZE"
tencrop = "TENCROP"
normalize = "NORMALIZE"
topilimage = "TOPILIMAGE"
totensor = "TOTENSOR"


# Parameters
brightness = "brightness"
contrast = "contrast"
saturation = "saturation"
hue = "hue"
num_output_channels = "num_output_channels"
transformation_matrix = "transformation_matrix"
fill = "fill"
padding_mode = "padding_mode"
degrees = "degrees"
translate = "translate"
scale = "scale"
shear = "shear"
resample = "resample"
fillcolor = "fillcolor"
pad_if_needed = "pad_if_needed"
ratio = "ratio"
interpolation = "interpolation"
expand = "expand"
center = "center"
vertical_flip = "vertical_flip"

# centercrop
size_dict = dict()
size_dict[_type] = [_int, _list]
size_dict[optional] = False

centercrop_params = dict()
centercrop_params[size] = deepcopy(size_dict)

centercrop_dict = dict()
centercrop_dict[path] = "torchvision.transforms.CenterCrop"
centercrop_dict[params] = deepcopy(centercrop_params)

transforms_dict[centercrop] = deepcopy(centercrop_dict)

#colorjitter
brightness_dict = dict()
brightness_dict[_type] = [_float]
brightness_dict[optional] = True
brightness_dict[default] = 0

contrast_dict = dict()
contrast_dict[_type] = [_float]
contrast_dict[optional] = True
contrast_dict[default] = 0

saturation_dict = dict()
saturation_dict[_type] = [_float]
saturation_dict[optional] = True
saturation_dict[default] = 0

hue_dict = dict()
hue_dict[_type] = [_float]
hue_dict[optional] = True
hue_dict[default] = 0

colorjitter_params = dict()
colorjitter_params[brightness] = brightness_dict
colorjitter_params[contrast] = contrast_dict
colorjitter_params[saturation] = saturation_dict
colorjitter_params[hue] = hue_dict

colorjitter_dict = dict()
colorjitter_dict[path] = "torchvision.transforms.ColorJitter"
colorjitter_dict[params] = deepcopy(colorjitter_params)

transforms_dict[colorjitter] = deepcopy(colorjitter_dict)

# FiveCrop
size_dict = dict()
size_dict[_type] = [_int, _list]
size_dict[optional] = False

fivecrop_params = dict()
fivecrop_params[size] = deepcopy(size_dict)

fivecrop_dict = dict()
fivecrop_dict[path] = "torchvision.transforms.FiveCrop"
fivecrop_dict[params] = deepcopy(fivecrop_params)

transforms_dict[fivecrop] = deepcopy(fivecrop_dict)

# Grayscale
num_output_channels_dict = dict()
num_output_channels_dict[_type] = [_int]
num_output_channels_dict[optional] = True
num_output_channels_dict[default] = 1

grayscale_params = dict()
grayscale_params[num_output_channels] = num_output_channels_dict

grayscale_dict = dict()
grayscale_dict[path] = "torchvision.transforms.Grayscale"
grayscale_dict[params] = grayscale_params

transforms_dict[grayscale] = grayscale_dict

# Linear Transformation
transformation_matrix_dict = dict()
transformation_matrix_dict[_type] = [_list]
transformation_matrix_dict[optional] = False

lineartransformation_params = dict()
lineartransformation_params[transformation_matrix] = transformation_matrix_dict

lineartransformation_dict = dict()
lineartransformation_dict[path] = "torch.nn.LinearTransformation"
lineartransformation_dict[params] = deepcopy(lineartransformation_params)

transforms_dict[lineartransformation] = deepcopy(lineartransformation_dict)

# Pad
padding_dict = dict()
padding_dict[_type] = [_int, _tuple]
padding_dict[optional] = False

fill_dict = dict()
fill_dict[_type] = [_int, _tuple]
fill_dict[optional] = True
fill_dict[default] = 0

padding_mode_dict = dict()
padding_mode_dict[_type] = [_str]
padding_mode_dict[optional] = True
padding_mode_dict[default] = "constant"

pad_params = dict()
pad_params[padding] = padding_dict
pad_params[fill] = fill_dict
pad_params[padding_mode] = padding_mode_dict

pad_dict = dict()
pad_dict[path] = "torchvision.transforms.Pad"
pad_dict[params] = deepcopy(pad_params)

transforms_dict[pad] = deepcopy(pad_dict)

# Random Affine
degrees_dict = dict()
degrees_dict[_type] = [_list, _float, _int]
degrees_dict[optional] = False

translate_dict = dict()
translate_dict[_type] = [_tuple]
translate_dict[optional] = True
translate_dict[default] = None

scale_dict = dict()
scale_dict[_type] = [_tuple]
scale_dict[optional] = True
scale_dict[default] = None

shear_dict = dict()
shear_dict[_type] = [_list, _float, _int]
shear_dict[optional] = True
shear_dict[default] = None

resample_dict = dict()
resample_dict[_type] = [_bool]
resample_dict[optional] = True
resample_dict[default] = False

fillcolor_dict = dict()
fillcolor_dict[_type] = [_int]
fillcolor_dict[optional] = True
fillcolor_dict[default] = 0

randomaffine_params = dict()
randomaffine_params[degrees] = degrees_dict
randomaffine_params[translate] = translate_dict
randomaffine_params[scale] = scale_dict
randomaffine_params[shear] = shear_dict
randomaffine_params[resample] = resample_dict
randomaffine_params[fillcolor] = fillcolor_dict

randomaffine_dict = dict()
randomaffine_dict[path] = "torchvision.transforms.RandomAffine"
randomaffine_dict[params] = deepcopy(randomaffine_params)

transforms_dict[randomaffine] = deepcopy(randomaffine_dict)

# Random Apply
# Cannot be implemented for now because the parameters to the function is a list of Transformation objects

# Random Choice
# Cannot be implemented for now because the parameters to the function is a list of Transformation objects

# Random crop
size_dict = dict()
size_dict[_type] = [_int, _list]
size_dict[optional] = False

padding_dict = dict()
padding_dict[_type] = [_int, _list]
padding_dict[optional] = True
padding_dict[default] = None

pad_if_needed_dict = dict()
pad_if_needed_dict[_type] = [_bool]
pad_if_needed_dict[optional] = True
pad_if_needed_dict[default] = False

fill_dict = dict()
fill_dict[_type] = [_int, _tuple]
fill_dict[optional] = True
fill_dict[default] = 0

padding_mode_dict = dict()
padding_mode_dict[_type] = [_str]
padding_mode_dict[optional] = True
padding_mode_dict[default] = "constant"

randomcrop_params = dict()
randomcrop_params[size] = size_dict
randomcrop_params[padding] = padding_dict
randomcrop_params[pad_if_needed] = pad_if_needed_dict
randomcrop_params[fill] = fill_dict
randomcrop_params[padding_mode] = padding_mode_dict

randomcrop_dict = dict()
randomcrop_dict[path] = "torchvision.transforms.RandomCrop"
randomcrop_dict[params] = deepcopy(randomcrop_params)

transforms_dict[randomcrop] = deepcopy(randomcrop_dict)

# Random Grayscale
p_dict = dict()
p_dict[_type] = [_int]
p_dict[optional] = True
p_dict[default] = 0.1

randomgrayscale_params = dict()
randomgrayscale_params[p] = p_dict

randomgrayscale_dict = dict()
randomgrayscale_dict[path] = "torchvision.transforms.RandomGrayscale"
randomgrayscale_dict[params] = deepcopy(randomgrayscale_params)

transforms_dict[randomgrayscale] = deepcopy(randomgrayscale_dict)

# Random Horizontal Flip
p_dict = dict()
p_dict[_type] = [_int]
p_dict[optional] = True
p_dict[default] = 0.1

randomhorizontalflip_params = dict()
randomhorizontalflip_params[p] = p_dict

randomhorizontalflip_dict = dict()
randomhorizontalflip_dict[path] = "torchvision.transforms.RandomHorizontalFlip"
randomhorizontalflip_dict[params] = deepcopy(randomhorizontalflip_params)

transforms_dict[randomhorizontalflip] = deepcopy(randomhorizontalflip_dict)

# Random Resized Crop
size_dict = dict()
size_dict[_type] = [_int, _list]
size_dict[optional] = False

scale_dict = dict()
scale_dict[_type] = [_tuple]
scale_dict[optional] = True
scale_dict[default] = [0.08, 1.0]

ratio_dict = dict()
ratio_dict[_type] = [_tuple]
ratio_dict[optional] = True
ratio_dict[default] = [0.75, 1.3333333333333333333]

interpolation_dict = dict()
interpolation_dict[_type] = [_int]
interpolation_dict[optional] = True
interpolation_dict[default] = 2

randomresizedcrop_params = dict()
randomresizedcrop_params[size] = size_dict
randomresizedcrop_params[scale] = scale_dict
randomresizedcrop_params[ratio] = ratio_dict
randomresizedcrop_params[interpolation] = interpolation_dict

randomresizedcrop_dict = dict()
randomresizedcrop_dict[path] = "torchvision.transforms.RandomResizedCrop"
randomresizedcrop_dict[params] = deepcopy(randomresizedcrop_params)

transforms_dict[randomresizedcrop] = deepcopy(randomresizedcrop_dict)


# Random Rotation
degrees_dict = dict()
degrees_dict[_type] = [_list, _float, _int]
degrees_dict[optional] = False

resample_dict = dict()
resample_dict[_type] = [_bool]
resample_dict[optional] = True
resample_dict[default] = False

expand_dict = dict()
expand_dict[_type] = [_bool]
expand_dict[optional] = True
expand_dict[default] = False

center_dict = dict()
center_dict[_type] = [_list]
center_dict[optional] = True
center_dict[default] = None

randomrotation_params = dict()
randomrotation_params[degrees] = degrees_dict
randomrotation_params[expand] = expand_dict
randomrotation_params[center] = center_dict

randomrotation_dict = dict()
randomrotation_dict[path] = "torchvision.transforms.RandomRotation"
randomrotation_dict[params] = deepcopy(randomrotation_params)

transforms_dict[randomrotation] = randomrotation_dict

# Random Vertical Flip
p_dict = dict()
p_dict[_type] = [_int]
p_dict[optional] = True
p_dict[default] = 0.1

randomverticalflip_params = dict()
randomverticalflip_params[p] = p_dict

randomverticalflip_dict = dict()
randomverticalflip_dict[path] = "torchvision.transforms.RandomVerticalFlip"
randomverticalflip_dict[params] = deepcopy(randomverticalflip_params)

transforms_dict[randomverticalflip] = deepcopy(randomverticalflip_params)

# Resize
size_dict = dict()
size_dict[_type] = [_int, _list]
size_dict[optional] = False

interpolation_dict = dict()
interpolation_dict[_type] = [_int]
interpolation_dict[optional] = True
interpolation_dict[default] = 2

resize_params = dict()
resize_params[size] = size_dict
resize_params[interpolation] = interpolation_dict

resize_dict = dict()
resize_dict[path] = "torchvision.transforms.Resize"
resize_dict[params] = deepcopy(resize_params)

transforms_dict[resize] = deepcopy(resize_dict)

# Ten Crop
size_dict = dict()
size_dict[_type] = [_int, _list]
size_dict[optional] = False

vertical_flip_dict = dict()
vertical_flip_dict[_type] = [_bool]
vertical_flip_dict[optional] = True
vertical_flip_dict[default] = False

tencrop_params = dict()
tencrop_params[size] = size_dict
tencrop_params[vertical_flip] = vertical_flip_dict

tencrop_dict = dict()
tencrop_dict[path] = "torchvision.transforms.TenCrop"
tencrop_dict[params] = tencrop_params

transforms_dict[tencrop] = tencrop_dict

# Testing part for the transforms layers
# Create a list with all the layer names in it
completed_transform = [
                        centercrop, colorjitter, fivecrop, grayscale, lineartransformation, pad, randomaffine,
                        randomapply, randomchoise, randomcrop, randomgrayscale, randomhorizontalflip,
                        randomorder, randomresizedcrop, randomrotation, randomsizedcrop, randomverticalflip, resize,
                        tencrop, normalize, topilimage, totensor
                      ]

test_dictionary(completed_object_list=completed_transform, input_dict=transforms_dict)

# This part contains the optimizers
# The parameters and functions are defined based on the documentation
# https://pytorch.org/docs/stable/optim.html#algorithms
adadelta = "ADADELTA"
adagrad = "ADAGRAD"
adam = "ADAM"
sparseadam = "SPARSEADAM"
adamax = "ADAMAX"
asgd = "ASGD"
lbfgs = "LBFGS"
rmsprop = "RMSPROP"
rprop = "RPROP"
sgd = "SGD"

#parameters
lr = "lr"
rho = "rho"
weight_decay = "weight_decay"
lr_decay = "lr_decay"
initial_accumulator_value = "initial_accumulator_value"
betas = "betas"
amsgrad = "amsgrad"
t0 = "t0"
max_iter = "max_iter"
max_eval = "max_eval"
tolerance_grad = "tolerance_grad"
tolerance_change = "tolerance_change"
history_size = "history_size"
centered = "centered"
etas = "etas"
step_sizes = "step_sizes"
dampening = "dampening"
nesterov = "nesterov"

optimizer_dict = dict()

# Adadelta
params_dict = dict()
params_dict[_type] = [_iterable]
params_dict[optional] = False

lr_dict = dict()
lr_dict[_type] = [_float]
lr_dict[optional] = True
lr_dict[default] = 1.0

rho_dict = dict()
rho_dict[_type] = [_float]
rho_dict[optional] = True
rho_dict[default] = 0.9

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-06

weight_decay_dict = dict()
weight_decay_dict[_type] = [_float]
weight_decay_dict[optional] = True
weight_decay_dict[default] = 0

adadelta_params = dict()
adadelta_params[params] = params_dict
adadelta_params[lr] = lr_dict
adadelta_params[rho] = rho_dict
adadelta_params[eps] = eps_dict
adadelta_params[weight_decay] = weight_decay_dict

adadelta_dict = dict()
adadelta_dict[path] = "torch.optim.Adadelta"
adadelta_dict[params] = deepcopy(adadelta_params)

optimizer_dict[adadelta] = deepcopy(adadelta_dict)

# Adagrad
params_dict = dict()
params_dict[_type] = [_iterable]
params_dict[optional] = False

lr_dict = dict()
lr_dict[_type] = [_float]
lr_dict[optional] = True
lr_dict[default] = 0.01

lr_decay_dict = dict()
lr_decay_dict[_type] = [_float]
lr_decay_dict[optional] = True
lr_decay_dict[default] = 0

weight_decay_dict = dict()
weight_decay_dict[_type] = [_float]
weight_decay_dict[optional] = True
weight_decay_dict[default] = 0

initial_accumulator_value_dict = dict()
initial_accumulator_value_dict[_type] = [_int]
initial_accumulator_value_dict[optional] = True
initial_accumulator_value_dict[default] = 0

adagrad_params = dict()
adagrad_params[params] = params_dict
adagrad_params[lr] = lr_dict
adagrad_params[lr_decay] = lr_decay_dict
adagrad_params[weight_decay] = weight_decay_dict
adagrad_params[initial_accumulator_value] = initial_accumulator_value_dict

adagrad_dict = dict()
adagrad_dict[path] = "torch.optim.Adagrad"
adagrad_dict[params] = deepcopy(adagrad_params)

optimizer_dict[adagrad] = deepcopy(adagrad_dict)

# Adam
params_dict = dict()
params_dict[_type] = [_iterable]
params_dict[optional] = False

lr_dict = dict()
lr_dict[_type] = [_float]
lr_dict[optional] = True
lr_dict[default] = 0.001

betas_dict = dict()
betas_dict[_type] = [_tuple]
betas_dict[optional] = True
betas_dict[default] = [0.9, 0.999]

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-08

weight_decay_dict = dict()
weight_decay_dict[_type] = [_float]
weight_decay_dict[optional] = True
weight_decay_dict[default] = 0

amsgrad_dict = dict()
amsgrad_dict[_type] = [_bool]
amsgrad_dict[optional] = True
amsgrad_dict[default] = False

adam_params = dict()
adam_params[params] = params_dict
adam_params[lr] = lr_dict
adam_params[betas] = betas_dict
adam_params[eps] = eps_dict
adam_params[weight_decay] = weight_decay_dict
adam_params[amsgrad] = amsgrad_dict

adam_dict = dict()
adam_dict[path] = "torch.optim.Adam"
adam_dict[params] = deepcopy(adam_params)

optimizer_dict[adam] = deepcopy(adam_dict)

# SparseAdam
params_dict = dict()
params_dict[_type] = [_iterable]
params_dict[optional] = False

lr_dict = dict()
lr_dict[_type] = [_float]
lr_dict[optional] = True
lr_dict[default] = 0.001

betas_dict = dict()
betas_dict[_type] = [_tuple]
betas_dict[optional] = True
betas_dict[default] = [0.9, 0.999]

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-08

weight_decay_dict = dict()
weight_decay_dict[_type] = [_float]
weight_decay_dict[optional] = True
weight_decay_dict[default] = 0

sparseadam_params = dict()
sparseadam_params[params] = params_dict
sparseadam_params[lr] = lr_dict
sparseadam_params[betas] = betas_dict
sparseadam_params[eps] = eps_dict

sparseadam_dict = dict()
sparseadam_dict[path] = "torch.optim.SparseAdam"
sparseadam_dict[params] = deepcopy(sparseadam_params)

optimizer_dict[sparseadam] = deepcopy(sparseadam_dict)

# Adamax
params_dict = dict()
params_dict[_type] = [_iterable]
params_dict[optional] = False

lr_dict = dict()
lr_dict[_type] = [_float]
lr_dict[optional] = True
lr_dict[default] = 0.002

betas_dict = dict()
betas_dict[_type] = [_tuple]
betas_dict[optional] = True
betas_dict[default] = [0.9, 0.999]

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-08

weight_decay_dict = dict()
weight_decay_dict[_type] = [_float]
weight_decay_dict[optional] = True
weight_decay_dict[default] = 0

sparseadam_params = dict()
sparseadam_params[params] = params_dict
sparseadam_params[lr] = lr_dict
sparseadam_params[betas] = betas_dict
sparseadam_params[eps] = eps_dict

sparseadam_dict = dict()
sparseadam_dict[path] = "torch.optim.Adamax"
sparseadam_dict[params] = deepcopy(sparseadam_params)

optimizer_dict[adamax] = deepcopy(sparseadam_dict)

# ASGD
params_dict = dict()
params_dict[_type] = [_iterable]
params_dict[optional] = False

lr_dict = dict()
lr_dict[_type] = [_float]
lr_dict[optional] = True
lr_dict[default] = 0.01

lambd_dict = dict()
lambd_dict[_type] = [_float]
lambd_dict[optional] = True
lambd_dict[default] = 0.0001

alpha_dict = dict()
alpha_dict[_type] = [_float]
alpha_dict[optional] = True
alpha_dict[default] = 0.75

t0_dict = dict()
t0_dict[_type] = [_float]
t0_dict[optional] = True
t0_dict[default] = 1000000.0

weight_decay_dict = dict()
weight_decay_dict[_type] = [_float]
weight_decay_dict[optional] = True
weight_decay_dict[default] = 0

asgd_params = dict()
asgd_params[params] = params_dict
asgd_params[lr] = lr_dict
asgd_params[lambd] = lambd_dict
asgd_params[alpha] = alpha_dict
asgd_params[t0] = t0_dict
asgd_params[weight_decay] = weight_decay_dict

asgd_dict = dict()
asgd_dict[path] = "torch.optim.ASGD"
asgd_dict[params] = asgd_params

optimizer_dict[asgd] = asgd_dict

# LBFGS
params_dict = dict()
params_dict[_type] = [_iterable]
params_dict[optional] = False

lr_dict = dict()
lr_dict[_type] = [_float]
lr_dict[optional] = True
lr_dict[default] = 1.0

max_iter_dict = dict()
max_iter_dict[_type] = [_int]
max_iter_dict[optional] = True
max_iter_dict[default] = 20

max_eval_dict = dict()
max_eval_dict[_type] = [_int]
max_eval_dict[optional] = True
max_eval_dict[default] = None

tolerance_grad_dict = dict()
tolerance_grad_dict[_type] = [_float]
tolerance_grad_dict[optional] = True
tolerance_grad_dict[default] = 1e-05

tolerance_change_dict = dict()
tolerance_change_dict[_type] = [_float]
tolerance_change_dict[optional] = True
tolerance_change_dict[default] = 1e-09


history_size_dict = dict()
history_size_dict[_type] = [_int]
history_size_dict[optional] = True
history_size_dict[default] = 100

lbfgs_params = dict()
lbfgs_params[params] = params_dict
lbfgs_params[lr] = lr_dict
lbfgs_params[max_iter] = max_iter_dict
lbfgs_params[max_eval] = max_eval_dict
lbfgs_params[tolerance_grad] = tolerance_grad_dict
lbfgs_params[tolerance_change] = tolerance_change_dict
lbfgs_params[history_size] = history_size_dict

lbfgs_dict = dict()
lbfgs_dict[path] = "torch.optim.LBFGS"
lbfgs_dict[params] = lbfgs_params

optimizer_dict[lbfgs] = lbfgs_dict

# RMSProp
params_dict = dict()
params_dict[_type] = [_iterable]
params_dict[optional] = False

lr_dict = dict()
lr_dict[_type] = [_float]
lr_dict[optional] = True
lr_dict[default] = 0.01

alpha_dict = dict()
alpha_dict[_type] = [_float]
alpha_dict[optional] = True
alpha_dict[default] = 0.99

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-08

weight_decay_dict = dict()
weight_decay_dict[_type] = [_float]
weight_decay_dict[optional] = True
weight_decay_dict[default] = 0

momentum_dict = dict()
momentum_dict[_type] = [_float]
momentum_dict[optional] = True
momentum_dict[default] = 0.0

centered_dict = dict()
centered_dict[_type] = [_bool]
centered_dict[optional] = True
centered_dict[default] = False

rmsprop_params = dict()
rmsprop_params[params] = params_dict
rmsprop_params[lr] = lr_dict
rmsprop_params[alpha] = alpha_dict
rmsprop_params[eps] = eps_dict
rmsprop_params[weight_decay] = weight_decay_dict
rmsprop_params[momentum] = momentum_dict
rmsprop_params[centered] = centered_dict

rmsprop_dict = dict()
rmsprop_dict[path] = "torch.optim.RMSprop"
rmsprop_dict[params] = deepcopy(rmsprop_params)

optimizer_dict[rmsprop] = deepcopy(rmsprop_dict)

# RPROP
params_dict = dict()
params_dict[_type] = [_iterable]
params_dict[optional] = False

lr_dict = dict()
lr_dict[_type] = [_float]
lr_dict[optional] = True
lr_dict[default] = 0.01

etas_dict = dict()
etas_dict[_type] = [_tuple]
etas_dict[optional] = True
etas_dict[default] = [0.5, 1.2]

step_sizes_dict = dict()
step_sizes_dict[_type] = [_tuple]
step_sizes_dict[optional] = True
step_sizes_dict[default] = [1e06, 50]

rprop_params = dict()
rprop_params[params] = params_dict
rprop_params[lr] = lr_dict
rprop_params[etas] = etas_dict
rprop_params[step_sizes] = step_sizes_dict

rprop_dict = dict()
rprop_dict[path] = "torch.optim.Rprop"
rprop_dict[params] = deepcopy(rprop_params)

optimizer_dict[rprop] = deepcopy(rprop_dict)

# SGD
params_dict = dict()
params_dict[_type] = [_iterable]
params_dict[optional] = False

lr_dict = dict()
lr_dict[_type] = [_float]
lr_dict[optional] = True
lr_dict[default] = 0.01

momentum_dict = dict()
momentum_dict[_type] = [_float]
momentum_dict[optional] = True
momentum_dict[default] = 0.0

dampening_dict = dict()
dampening_dict[_type] = [_float]
dampening_dict[optional] = True
dampening_dict[default] = 0.0

weight_decay_dict = dict()
weight_decay_dict[_type] = [_float]
weight_decay_dict[optional] = True
weight_decay_dict[default] = 0

nesterov_dict = dict()
nesterov_dict[_type] = [_bool]
nesterov_dict[optional] = True
nesterov_dict[default] = False

sgd_params = dict()
sgd_params[params] = params_dict
sgd_params[lr] = lr_dict
sgd_params[momentum] = momentum_dict
sgd_params[dampening] = dampening_dict
sgd_params[weight_decay] = weight_decay_dict
sgd_params[nesterov] = nesterov_dict

sgd_dict = dict()
sgd_dict[path] = "torch.optim.SGD"
sgd_dict[params] = deepcopy(sgd_params)

optimizer_dict[sgd] = deepcopy(sgd_dict)

completed_optimizers = [adadelta, adagrad, adam, sparseadam, adamax, asgd, lbfgs, rmsprop, rprop, sgd]
test_dictionary(completed_object_list=completed_optimizers, input_dict=optimizer_dict)

# The following part implements the loss functions based on the pytorch documentation
# https://pytorch.org/docs/stable/nn.html#loss-functions
l1loss = "L1LOSS"
mseloss = "MSELOSS"
crossentropyloss = "CROSSENTROPYLOSS"
nllloss = "NLLLOSS"
poissonnllloss = "POISSONNLLLOSS"
kldivloss = "KLDIVLOSS"
bceloss = "BCELOSS"
bcewithlogitsloss = "BCEWITHLOGITSLOSS"
marginrankingloss = "MARGINRANKINGLOSS"
hingeembeddingloss = "HINGEEMBEDDINGLOSS"
multilabelmarginloss = "MULTILABELMARGINLOSS"
smoothl1loss = "SMOOTHL1LOSS"
softmarginloss = "SOFTMARGINLOSS"
multilabelsoftmarginloss = "MULTILABELSOFTMARGINLOSS"
cosineembeddingloss = "COSINEEMBEDDINGLOSS"
multimarginloss = "MULTIMARGINLOSS"
tripletmarginloss = "TRIPLETMARGINLOSS"

# parameters
size_average = "size_average"
reduce = "reduce"
reduction = "reduction"
weight = "weight"
ignore_index = "ignore_index"
log_input = "log_input"
full = "full"
pos_weight = "pos_weight"
margin = "margin"

loss_dict = dict()

# L1 Loss
size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

l1loss_params = dict()
l1loss_params[size_average] = size_average_dict
l1loss_params[reduce] = reduce_dict
l1loss_params[reduction] = reduction_dict

l1loss_dict = dict()
l1loss_dict[path] = "torch.nn.L1Loss"
l1loss_dict[params] = deepcopy(l1loss_params)

loss_dict[l1loss] = deepcopy(l1loss_dict)

# MSE Loss
size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

mseloss_params = dict()
mseloss_params[size_average] = size_average_dict
mseloss_params[reduce] = reduce_dict
mseloss_params[reduction] = reduction_dict

mseloss_dict = dict()
mseloss_dict[path] = "torch.nn.MSELoss"
mseloss_dict[params] = deepcopy(mseloss_params)

loss_dict[mseloss] = deepcopy(mseloss_dict)

# Cross Entropy Loss
weight_dict = dict()
weight_dict[_type] = [_tensor]
weight_dict[optional] = True
weight_dict[default] = None

size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

ignore_index_dict = dict()
ignore_index_dict[_type] = [_int]
ignore_index_dict[optional] = True
ignore_index_dict[default] = -100

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

crossentropyloss_params = dict()
crossentropyloss_params[weight] = weight_dict
crossentropyloss_params[size_average] = size_average_dict
crossentropyloss_params[ignore_index] = ignore_index_dict
crossentropyloss_params[reduce] = reduce_dict
crossentropyloss_params[reduction] = reduction_dict

crossentropyloss_dict = dict()
crossentropyloss_dict[path] = "torch.nn.CrossEntropyLoss"
crossentropyloss_dict[params] = deepcopy(crossentropyloss_params)

loss_dict[crossentropyloss] = deepcopy(crossentropyloss_dict)

# NLL Loss
weight_dict = dict()
weight_dict[_type] = [_tensor]
weight_dict[optional] = True
weight_dict[default] = None

size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

ignore_index_dict = dict()
ignore_index_dict[_type] = [_int]
ignore_index_dict[optional] = True
ignore_index_dict[default] = -100

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

nllloss_params = dict()
nllloss_params[weight] = weight_dict
nllloss_params[size_average] = size_average_dict
nllloss_params[ignore_index] = ignore_index_dict
nllloss_params[reduce] = reduce_dict
nllloss_params[reduction] = reduction_dict

nllloss_dict = dict()
nllloss_dict[path] = "torch.nn.NLLLoss"
nllloss_dict[params] = deepcopy(nllloss_params)

loss_dict[nllloss] = deepcopy(nllloss_dict)

# Poisson NLL Loss
log_input_dict = dict()
log_input_dict[_type] = [_bool]
log_input_dict[optional] = True
log_input_dict[default] = True

full_dict = dict()
full_dict[_type] = [_bool]
full_dict[optional] = True
full_dict[default] = False

size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-08

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

poissonnllloss_params = dict()
poissonnllloss_params[log_input] = log_input_dict
poissonnllloss_params[full] = full_dict
poissonnllloss_params[size_average] = size_average_dict
poissonnllloss_params[eps] = eps_dict
poissonnllloss_params[reduce] = reduce_dict
poissonnllloss_params[reduction] = reduction_dict

poissonnllloss_dict = dict()
poissonnllloss_dict[path] = "torch.nn.PoissonNLLLoss"
poissonnllloss_dict[params] = deepcopy(poissonnllloss_params)

loss_dict[poissonnllloss] = deepcopy(poissonnllloss_dict)

# Kullback-Leibler Divergence Loss
size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

kldivloss_params = dict()
kldivloss_params[size_average] = size_average_dict
kldivloss_params[reduce] = reduce_dict
kldivloss_params[reduction] = reduction_dict

kldivloss_dict = dict()
kldivloss_dict[path] = "torch.nn.KLDivLoss"
kldivloss_dict[params] = deepcopy(kldivloss_params)

loss_dict[kldivloss] = deepcopy(kldivloss_dict)

# Binary Cross Entropy Loss
weight_dict = dict()
weight_dict[_type] = [_tensor]
weight_dict[optional] = True
weight_dict[default] = None

size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

bceloss_params = dict()
bceloss_params[weight] = weight_dict
bceloss_params[size_average] = size_average_dict
bceloss_params[reduce] = reduce_dict
bceloss_params[reduction] = reduction_dict

bceloss_dict = dict()
bceloss_dict[path] = "torch.nn.BCELoss"
bceloss_dict[params] = deepcopy(bceloss_params)

loss_dict[bceloss] = deepcopy(bceloss_dict)

# Binary Cross Entropy Loss with Logits
weight_dict = dict()
weight_dict[_type] = [_tensor]
weight_dict[optional] = True
weight_dict[default] = None

size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

pos_weight_dict = dict()
pos_weight_dict[_type] = [_list]
pos_weight_dict[optional] = True
pos_weight_dict[default] = None


bcewithlogitsloss_params = dict()
bcewithlogitsloss_params[weight] = weight_dict
bcewithlogitsloss_params[size_average] = size_average_dict
bcewithlogitsloss_params[reduce] = reduce_dict
bcewithlogitsloss_params[reduction] = reduction_dict
bcewithlogitsloss_params[pos_weight] = pos_weight_dict

bcewithlogitsloss_dict = dict()
bcewithlogitsloss_dict[path] = "torch.nn.BCEWithLogitsLoss"
bcewithlogitsloss_dict[params] = deepcopy(bcewithlogitsloss_params)

loss_dict[bcewithlogitsloss] = deepcopy(bcewithlogitsloss_dict)

# Margin Ranking Loss
margin_dict = dict()
margin_dict[_type] = [_float]
margin_dict[optional] = True
margin_dict[default] = 0.0

size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

marginrankingloss_params = dict()
marginrankingloss_params[margin] = margin_dict
marginrankingloss_params[size_average] = size_average_dict
marginrankingloss_params[reduce] = reduce_dict
marginrankingloss_params[reduction] = reduction_dict

marginrankingloss_dict = dict()
marginrankingloss_dict[path] = "torch.nn.MarginRankingLoss"
marginrankingloss_dict[params] = deepcopy(marginrankingloss_params)

loss_dict[marginrankingloss] = deepcopy(marginrankingloss_dict)

# Hinge Embedding Loss
margin_dict = dict()
margin_dict[_type] = [_float]
margin_dict[optional] = True
margin_dict[default] = 1.0

size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

hingeembeddingloss_params = dict()
hingeembeddingloss_params[margin] = margin_dict
hingeembeddingloss_params[size_average] = size_average_dict
hingeembeddingloss_params[reduce] = reduce_dict
hingeembeddingloss_params[reduction] = reduction_dict

hingeembeddingloss_dict = dict()
hingeembeddingloss_dict[path] = "torch.nn.HingeEmbeddingLoss"
hingeembeddingloss_dict[params] = deepcopy(hingeembeddingloss_params)

loss_dict[hingeembeddingloss] = deepcopy(hingeembeddingloss_dict)

# Multilabel Margin Loss
size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

multilabelmarginloss_params = dict()
multilabelmarginloss_params[size_average] = size_average_dict
multilabelmarginloss_params[reduce] = reduce_dict
multilabelmarginloss_params[reduction] = reduction_dict

multilabelmarginloss_dict = dict()
multilabelmarginloss_dict[path] = "torch.nn.MultiLabelMarginLoss"
multilabelmarginloss_dict[params] = deepcopy(multilabelmarginloss_params)

loss_dict[multilabelmarginloss] = deepcopy(multilabelmarginloss_dict)

# Smooth L1 Loss
size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

smoothl1loss_params = dict()
smoothl1loss_params[size_average] = size_average_dict
smoothl1loss_params[reduce] = reduce_dict
smoothl1loss_params[reduction] = reduction_dict

smooothl1loss_dict = dict()
smooothl1loss_dict[path] = "torch.nn.SmoothL1Loss"
smooothl1loss_dict[params] = deepcopy(smoothl1loss_params)

loss_dict[smoothl1loss] = deepcopy(smooothl1loss_dict)

# Soft Margin Loss
size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

softmarginloss_params = dict()
softmarginloss_params[size_average] = size_average_dict
softmarginloss_params[reduce] = reduce_dict
softmarginloss_params[reduction] = reduction_dict

softmarginloss_dict = dict()
softmarginloss_dict[path] = "torch.nn.SoftMarginLoss"
softmarginloss_dict[params] = deepcopy(softmarginloss_params)

loss_dict[softmarginloss] = deepcopy(softmarginloss_dict)

# Multilable Soft Margin Loss
weight_dict = dict()
weight_dict[_type] = [_tensor]
weight_dict[optional] = True
weight_dict[default] = None

size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

multilabelsoftmarginloss_params = dict()
multilabelsoftmarginloss_params[weight] = weight_dict
multilabelsoftmarginloss_params[size_average] = size_average_dict
multilabelsoftmarginloss_params[reduce] = reduce_dict
multilabelsoftmarginloss_params[reduction] = reduction_dict

multilabelsoftmarginloss_dict = dict()
multilabelsoftmarginloss_dict[path] = "torch.nn.MultiLabelSoftMarginLoss"
multilabelsoftmarginloss_dict[params] = deepcopy(multilabelsoftmarginloss_params)

loss_dict[multilabelsoftmarginloss] = deepcopy(multilabelsoftmarginloss_dict)

# Cosine Embedding Loss
margin_dict = dict()
margin_dict[_type] = [_float]
margin_dict[optional] = True
margin_dict[default] = 1.0

size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

cosineembeddingloss_params = dict()
cosineembeddingloss_params[margin] = weight_dict
cosineembeddingloss_params[size_average] = size_average_dict
cosineembeddingloss_params[reduce] = reduce_dict
cosineembeddingloss_params[reduction] = reduction_dict

cosineembeddingloss_params_dict = dict()
cosineembeddingloss_params_dict[path] = "torch.nn.CosineEmbeddingLoss"
cosineembeddingloss_params_dict[params] = deepcopy(cosineembeddingloss_params)

loss_dict[cosineembeddingloss] = deepcopy(cosineembeddingloss_params_dict)

# Multi Margin Loss
p_dict = dict()
p_dict[_type] = [_int]
p_dict[optional] = True
p_dict[default] = 1

margin_dict = dict()
margin_dict[_type] = [_float]
margin_dict[optional] = True
margin_dict[default] = 1.0

weight_dict = dict()
weight_dict[_type] = [_tensor]
weight_dict[optional] = True
weight_dict[default] = None

size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

multimarginloss_params = dict()
multimarginloss_params[p] = p_dict
multimarginloss_params[weight] = weight_dict
multimarginloss_params[margin] = weight_dict
multimarginloss_params[size_average] = size_average_dict
multimarginloss_params[reduce] = reduce_dict
multimarginloss_params[reduction] = reduction_dict

multimarginloss_dict = dict()
multimarginloss_dict[path] = "torch.nn.MultiMarginLoss"
multimarginloss_dict[params] = deepcopy(multimarginloss_params)

loss_dict[multimarginloss] = deepcopy(multimarginloss_dict)

# Triplet Margin Loss
margin_dict = dict()
margin_dict[_type] = [_float]
margin_dict[optional] = True
margin_dict[default] = 1.0

p_dict = dict()
p_dict[_type] = [_int]
p_dict[optional] = True
p_dict[default] = 1

eps_dict = dict()
eps_dict[_type] = [_float]
eps_dict[optional] = True
eps_dict[default] = 1e-08

swap_dict = dict()
swap_dict[_type] = [_bool]
swap_dict[optional] = True
swap_dict[default] = False

size_average_dict = dict()
size_average_dict[_type] = [_bool]
size_average_dict[optional] = True
size_average_dict[default] = True

reduce_dict = dict()
reduce_dict[_type] = [_bool]
reduce_dict[optional] = True
reduce_dict[default] = True

reduction_dict = dict()
reduction_dict[_type] = [_str]
reduction_dict[optional] = True
reduction_dict[default] = "elementwise_mean"

tripletmarginloss_params = dict()
tripletmarginloss_params[p] = p_dict
tripletmarginloss_params[weight] = weight_dict
tripletmarginloss_params[margin] = weight_dict
tripletmarginloss_params[size_average] = size_average_dict
tripletmarginloss_params[reduce] = reduce_dict
tripletmarginloss_params[reduction] = reduction_dict

tripletmarginloss_dict = dict()
tripletmarginloss_dict[path] = "torch.nn.TripletMarginLoss"
tripletmarginloss_dict[params] = deepcopy(tripletmarginloss_params)

loss_dict[tripletmarginloss] = deepcopy(tripletmarginloss_dict)

completed_loss_functions = [
                            l1loss, mseloss, crossentropyloss, nllloss, poissonnllloss, kldivloss, bceloss,
                            bcewithlogitsloss, marginrankingloss, hingeembeddingloss, multilabelmarginloss,
                            smoothl1loss, softmarginloss, multilabelsoftmarginloss, cosineembeddingloss,
                            multimarginloss, tripletmarginloss
                           ]

test_dictionary(completed_loss_functions, loss_dict)

# Print the current working directory and write the dictionary to JSON file
layers = "layers"
transforms = "transforms"
optimizers = "optimizers"
loss_functions = "loss_functions"

framework_dict = dict()
framework_dict[layers] = layers_dict
framework_dict[transforms] = transforms_dict
framework_dict[optimizers] = optimizer_dict
framework_dict[loss_functions] = loss_dict

cwd = os.getcwd()
print(cwd)
with open('mappings_torch.json', 'w') as fp:
    json.dump(framework_dict, fp)

