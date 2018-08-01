from copy import deepcopy
import json

# This code has been created with the pytorch documentation from "https://pytorch.org/docs/stable/nn.html"
# Random samples parameter not included in FractionalMaxPool2D as the documentation does not specify the type

# The final dictionary to be dumped to JSON
from numpy.distutils.system_info import openblas_info

layers_dict = dict()

# The different strings present in the dictionaries are declared below
optional = "optional"
_type = "type"
path = "path"
_int = "int"
_tuple = "tuple"
_bool = "bool"
_float = "float"
_list = "list"
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

layers_dict[hardtanh] = hardtanh_dict

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
prelu_dict[params] = prelu_params

layers_dict[prelu] = prelu_dict

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
relu6_dict[params] = relu6_params

layers_dict[relu6] = relu6_dict

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
rrelu_dict[params] = rrelu_params

layers_dict[rrelu] = rrelu_dict

# SELU
inplace_dict = dict()
inplace_dict[_type] = [_bool]
inplace_dict[optional] = True
inplace_dict[default] = False

selu_params = dict()
selu_params[inplace] = inplace_dict

selu_dict = dict()
selu_dict[path] = "torch.nn.SELU"
selu_dict[params] = selu_params

layers_dict[selu] = selu_dict

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
softshrink_dict[params] = softshrink_params

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

layers_dict[threshold] = threshold_dict

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
groupnorm_dict[params] = groupnorm_params

layers_dict[groupnorm] = groupnorm_dict

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
dropout_dict[params] = dropout_params

layers_dict[dropout] = dropout_dict

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
dropout2d_dict[params] = dropout2d_params

layers_dict[dropout2d] = dropout2d_dict

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
dropout3d_dict[params] = dropout3d_params

layers_dict[dropout3d] = dropout3d_dict

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
alphadropout_dict[params] = alphadropout_params

layers_dict[alphadropout] = alphadropout_dict


# Print the current working directory and write the dictionary to JSON file
import os
cwd = os.getcwd()
print(cwd)
with open('torch_params.json', 'w') as fp:
    json.dump(layers_dict, fp)

print(layers_dict)

# Create a list with all the layer names in it
list_layer_names = []

# This part is for the testing of the entered layers
# Tests are
# 1. All layers should have unique paths
# 2. Default parameters should have a value associated with it
# 3. Compulsory parameters should not have a value along with it
# 4. The types possible should be a list
# 5. Verify that all defined layers are present within the layers_dict


completed_layers =\
                    [conv1d, conv2d, conv3d, transpose1d, transpose2d, transpose3d, unfold, fold, maxpool1d, maxpool2d,
                    maxpool3d, maxunpool1d, maxunpool2d, maxunpool3d, avgpool1d, avgpool2d, avgpool3d, fractionalmaxpool2d,
                    lppool1d, lppool2d, adaptivemaxpool1d, adaptivemaxpool2d, adaptivemaxpool3d, adaptiveavgpool1d,
                    reflectionpad1d, reflectionpad2d, replicationpad1d, replicationpad2d, replicationpad3d, zeropad2d,
                    constantpad1d, constantpad2d, constantpad3d, adaptiveavgpool2d, adaptiveavgpool3d,  elu, hardshrink,
                    hardtanh, leakyrelu, logsigmoid, prelu, relu, relu6, rrelu, selu, sigmoid, softplus, softshrink,
                    softsign, tanh, tanhshrink, threshold, softmin, softmax, softmax2d, logsoftmax,
                    adaptivelogsoftmaxwithloss, batchnorm1d, batchnorm2d, batchnorm3d, groupnorm, instancenorm1d,
                    instancenorm2d, instancenorm3d, layernorm, localresponsenorm, linear, bilinear, dropout, dropout2d,
                    dropout3d, alphadropout
                    ]

layer_paths = []
for layer_name in completed_layers:

    if layers_dict.get(layer_name, None) is None:
        print('Missing entry for ' + layer_name)
        continue

    curr_layer_path = layers_dict.get(layer_name, None).get(path)

    if curr_layer_path in layer_paths:
        print('Path duplicate found for path:' + curr_layer_path)

    else:
        layer_paths.append(curr_layer_path)



    layer_params_dict = layers_dict.get(layer_name).get(params, None)

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
