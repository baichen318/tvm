"""
HeteroCL Tutorial : LeNet Inference
===================================

**Author**: Yi-Hsiang Lai (seanlatias@github)

Build the LeNet inference model by using hlib. In this tutorial, we demonstrate
how we can apply inference-time quantization to an existing model.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'heterocl'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'hlib', 'python'))
import heterocl as hcl
import hlib
import numpy as np

def build_lenet(input_image):
    # first conv
    tanh1 = hlib.nn.tanh(input_image)
    #flat = hlib.nn.flatten(conv1)

    return tanh1

import mxnet as mx
# download pretrained lenet model
# mx.gluon.utils.download('https://gist.githubusercontent.com/Huyuwei/dc00ce83f537914c64a204133d23b019/raw/79af41e7c8ba9120ea7f35fb1d0484b65bccd54f/lenet-0010.params')
# mx.gluon.utils.download('https://gist.githubusercontent.com/Huyuwei/dc00ce83f537914c64a204133d23b019/raw/79af41e7c8ba9120ea7f35fb1d0484b65bccd54f/lenet-symbol.json')
sym, arg_params, aux_params = mx.model.load_checkpoint('lenet', 10)
# get weights
weight_conv1_np = arg_params['convolution0_weight'].asnumpy()
#print(weight_conv1_np)
#print(weight_conv1_np.shape)
# Define the quantized data type and run the inference
qtype1 = hcl.Fixed(16, 14)
qtype2 = hcl.Fixed(16, 14)
correct_sum = 0
batch_size = 1000
mnist = mx.test_utils.get_mnist()

def build_lenet_inf(batch_size=batch_size, target=None):
    # set up input/output placeholders
    input_image = hcl.placeholder((batch_size, 1, 28, 28), "input_image")
    #weight_conv1 = hcl.placeholder((20, 1, 5, 5), "weight_conv1", qtype1)
    scheme = hcl.create_scheme(
            [input_image], build_lenet)
    s = hcl.create_schedule_from_scheme(scheme)
    #print(hcl.lower(s))
    return hcl.build(s, target=target)

f = build_lenet_inf()

weight_conv1_hcl = hcl.asarray(weight_conv1_np, dtype=qtype1)
for i in range(10000 // batch_size):
    label = mnist['test_label'][i*batch_size:(i+1)*batch_size]
    input_image_np = mnist['test_data'][i*batch_size:(i+1)*batch_size]
    input_image_hcl = hcl.asarray(input_image_np)
    output_hcl = hcl.asarray(np.zeros((batch_size, 1, 28, 28)))
    f(input_image_hcl, output_hcl)
    output_np = output_hcl.asnumpy()
    print(output_np)



