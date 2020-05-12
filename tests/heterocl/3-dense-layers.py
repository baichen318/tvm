import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'heterocl'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'hlib', 'python'))

import heterocl as hcl
import hlib
import numpy as np

###############################################################################
# Initially we let every operation to be a floating-point operation
hcl.init(hcl.Float())

###############################################################################
# You can define your own layer without using the one provided in hlib
###############################################################################
# The main function for build the LeNet inference model
def build_lenet(input_data, weight_fc1, weight_fc2, weight_fc3):
    fc1 = hlib.nn.dense(input_data, weight_fc1)
    tanh1 = hlib.nn.tanh(fc1, "tanh1")
    fc2 = hlib.nn.dense(tanh1, weight_fc2)
    tanh2 = hlib.nn.tanh(fc2, "tanh2")
    fc3 = hlib.nn.dense(tanh2, weight_fc3)
    tanh3 = hlib.nn.tanh(fc3, "tanh3")
    result = hlib.nn.flatten(tanh3)
    # second fc
    # loss
    return result

###############################################################################
# Download the dataset from mxnet
#import mxnet as mx
# download pretrained lenet model
#mx.gluon.utils.download('https://gist.githubusercontent.com/Huyuwei/dc00ce83f537914c64a204133d23b019/raw/79af41e7c8ba9120ea7f35fb1d0484b65bccd54f/lenet-0010.params')
#mx.gluon.utils.download('https://gist.githubusercontent.com/Huyuwei/dc00ce83f537914c64a204133d23b019/raw/79af41e7c8ba9120ea7f35fb1d0484b65bccd54f/lenet-symbol.json')
#sym, arg_params, aux_params = mx.model.load_checkpoint('lenet', 10)
# get weights
#weight_conv1_np = arg_params['convolution0_weight'].asnumpy()
#weight_conv2_np = arg_params['convolution1_weight'].asnumpy()
weight_fc1_np = np.array([[0.93511265, 0.6649714 , 0.7263339 , 0.95472297, 0.65521839],
                       [0.05118694, 0.87881144, 0.96832627, 0.30732032, 0.15651566],
                       [0.44775005, 0.9806793 , 0.33559141, 0.50728622, 0.1845889 ],
                       [0.95641498, 0.85400729, 0.41556586, 0.10544507, 0.22375721],
                       [0.19135459, 0.99168525, 0.91864897, 0.79313401, 0.22158455]])

weight_fc2_np = np.array([[0.56572363, 0.59494232, 0.70425923, 0.08756663, 0.59860402],
                       [0.72610163, 0.70115349, 0.43475407, 0.51204054, 0.55115578],
                       [0.68040191, 0.70410196, 0.98877277, 0.08153342, 0.00593014],
                       [0.93789813, 0.30217619, 0.23484365, 0.10977669, 0.46587137],
                       [0.18587903, 0.79178911, 0.23977058, 0.00416712, 0.92055303]])

weight_fc3_np = np.array([[0.83065922, 0.21550846, 0.3823437 , 0.16557572, 0.68575093]])
#weight_fc2_np = arg_params['fullyconnected1_weight'].asnumpy()

###############################################################################
# Define the quantized data type and run the inference
qtype1 = hcl.Fixed(16, 14)
qtype2 = hcl.Fixed(16, 14)
#correct_sum = 0
#batch_size = 1000
#mnist = mx.test_utils.get_mnist()

###############################################################################
# In this example, we quantize the weights to `qtype1` and the activations to
# `qtype2`. To quantize the placeholders, simply specify the `dtype` field. For
# the internal tensors, we use `hcl.quantize` API.

def build_lenet_inf(batch_size=1, target=None):
    # set up input/output placeholders
    input_data = hcl.placeholder((1, 5), "input_data")
    weight_fc1 = hcl.placeholder((5, 5), "weight_fc1")
    weight_fc2 = hcl.placeholder((5, 5), "weight_fc2")
    weight_fc3 = hcl.placeholder((1, 5), "weight_fc3")
    #weight_fc2 = hcl.placeholder((10, 500), "weight_fc2", qtype1)
    #lenet = hcl.placeholder((batch_size, 10), "lenet")
    # create a quantization scheme
    scheme = hcl.create_scheme(
            [input_data,
             weight_fc1, weight_fc2, weight_fc3], build_lenet)
    # quantize the three activation layers
    scheme.quantize([build_lenet.tanh1, build_lenet.tanh2, build_lenet.tanh3], qtype2)
    s = hcl.create_schedule_from_scheme(scheme)
    print(hcl.lower(s))
    return hcl.build(s, target=target)
f = build_lenet_inf()

###############################################################################
# Prepare the numpy arrays for testing. Remember that we need to set the input
# tensors with the same type as the placeholders
#weight_conv1_hcl = hcl.asarray(weight_conv1_np, dtype=qtype1)
#weight_conv2_hcl = hcl.asarray(weight_conv2_np, dtype=qtype1)
weight_fc1_hcl = hcl.asarray(weight_fc1_np)
weight_fc2_hcl = hcl.asarray(weight_fc2_np)
weight_fc3_hcl = hcl.asarray(weight_fc3_np)
#weight_fc2_hcl = hcl.asarray(weight_fc2_np, dtype=qtype1)
#x = weight_conv1_hcl.asnumpy()
#print(x)

    #label = mnist['test_label'][i*batch_size:(i+1)*batch_size]
    #input_image_np = mnist['test_data'][i*batch_size:(i+1)*batch_size]
input_data_np = np.array([[1, 1, 1, 1, 1]])
input_data_hcl = hcl.asarray(input_data_np)
output_hcl = hcl.asarray(np.zeros((1, 1)))
f(input_data_hcl,
        weight_fc1_hcl, weight_fc2_hcl, weight_fc3_hcl, output_hcl)
print(output_hcl)