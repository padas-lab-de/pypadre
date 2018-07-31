from copy import deepcopy
import json

# The final dictionary to be dumped to JSON
layers_dict = dict()

# The different strings present in the dictionaries are declared below
optional = "optional"
_type = "type"
path = "path"
_int = "int"
_tuple = "tuple"
_bool = "bool"
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

# The different parameters for the layers are declared below
in_channels = "in_channels"
out_channels = "out_channels"
kernel_size = "kernel_size"
stride = "stride"
padding = "padding"
dilation = "dilation"
groups = "groups"
bias = "bias"


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
bias_dict[_type] = _bool
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
bias_dict[_type] = _bool
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
bias_dict[_type] = _bool
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
bias_dict[_type] = _bool
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
bias_dict[_type] = _bool
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
bias_dict[_type] = _bool
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
fold_params[kernel_size] = kernel_size_dict
fold_params[stride] = stride_dict
fold_params[padding] = padding_dict
fold_params[dilation] = dilation_dict

fold_dict = dict()
fold_dict[path] = "torch.nn.Fold"
fold_dict[params] = deepcopy(fold_params)

layers_dict[fold] = deepcopy(fold_dict)

# Print the current working directory and write the dictionary to JSON file
import os
cwd = os.getcwd()
print(cwd)
with open('torch_params.json', 'w') as fp:
    json.dump(layers_dict, fp)

print(layers_dict)














