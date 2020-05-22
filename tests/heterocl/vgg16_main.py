import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'heterocl'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
    'hlib', 'python'))
import heterocl as hcl
import hlib
import numpy as np
from PIL import Image

###############################################################################
# Initially we let every operation to be a floating-point operation
hcl.init(hcl.Float())

def build_model(input_image, weight_conv1, weight_conv2, weight_conv3, weight_conv4,
                weight_conv5, weight_conv6, weight_conv7, weight_conv8, weight_conv9,
                weight_conv10, weight_conv11, weight_conv12, weight_conv13, weight_fc1,
                weight_fc2, weight_fc3, vgg):
    conv1 = hlib.nn.conv2d_nchw(input_image, weight_conv1, padding=[[1,1],[1,1]])
    tanh1 = hlib.nn.tanh(conv1, "tanh1")
    conv2 = hlib.nn.conv2d_nchw(tanh1, weight_conv2)
    tanh2 = hlib.nn.tanh(conv2, "tanh2")
    pool1 = hlib.nn.max_pool(tanh2, kernel=(2, 2), stride=(2, 2), padding=[[1,1],[1,1]])
    conv3 = hlib.nn.conv2d_nchw(pool1, weight_conv3)
    tanh3 = hlib.nn.tanh(conv3, "tanh3")
    conv4 = hlib.nn.conv2d_nchw(tanh3, weight_conv4)
    tanh4 = hlib.nn.tanh(conv4, "tanh4")
    pool2 = hlib.nn.max_pool(tanh4, kernel=(2,2), stride=(2,2), padding=[[1,1],[1,1]])
    conv5 = hlib.nn.conv2d_nchw(pool2, weight_conv5)
    tanh5 = hlib.nn.tanh(conv5, "tanh5")
    conv6 = hlib.nn.conv2d_nchw(tanh5, weight_conv6)
    tanh6 = hlib.nn.tanh(conv6, "tanh6")
    conv7 = hlib.nn.conv2d_nchw(tanh6, weight_conv7)
    tanh7 = hlib.nn.tanh(conv7, "tanh7")
    pool3 = hlib.nn.max_pool(tanh7, kernel=(2,2), stride=(2,2), padding=[[1,1],[1,1]])
    conv5 = hlib.nn.conv2d_nchw(pool3, weight_conv8)
    tanh5 = hlib.nn.tanh(conv5, "tanh5")
    conv6 = hlib.nn.conv2d_nchw(tanh5, weight_conv9)
    tanh6 = hlib.nn.tanh(conv6, "tanh6")
    conv7 = hlib.nn.conv2d_nchw(tanh6, weight_conv10)
    tanh7 = hlib.nn.tanh(conv7, "tanh7")
    pool4 = hlib.nn.max_pool(tanh7, kernel=(2,2), stride=(2,2), padding=[[1,1],[1,1]])
    conv5 = hlib.nn.conv2d_nchw(pool4, weight_conv11)
    tanh5 = hlib.nn.tanh(conv5, "tanh5")
    conv6 = hlib.nn.conv2d_nchw(tanh5, weight_conv12)
    tanh6 = hlib.nn.tanh(conv6, "tanh6")
    conv7 = hlib.nn.conv2d_nchw(tanh6, weight_conv13)
    tanh7 = hlib.nn.tanh(conv7, "tanh7")
    pool5 = hlib.nn.max_pool(tanh7, kernel=(2,2), stride=(2,2), padding=[[1,1],[1,1]])
    pool6 = hlib.nn.max_pool(pool5, kernel=(2,2), stride=(2,2), padding=[[1,1],[1,1]])
    flat1 = hlib.nn.flatten(pool6)
    dense1 = hlib.nn.dense(flat1, weight_fc1)
    tanh8 = hlib.nn.tanh(dense1, "tanh8")
    dense2 = hlib.nn.dense(tanh8, weight_fc2)
    tanh9 = hlib.nn.tanh(dense2, "tanh9")
    dense3 = hlib.nn.dense(tanh9, weight_fc3)
    result = hlib.nn.softmax(vgg, dense3)

    return result

qtype1 = hcl.Fixed(16, 14)
qtype2 = hcl.Fixed(16, 14)
correct_sum = 0
batch_size = 1000


def build_model_inf(batch_size=batch_size, target=None):
    # set up input/output placeholders
    input_image = hcl.placeholder((batch_size, 3, 224, 224), "input_image")
    weight_conv1 = hcl.placeholder((64, 3, 3, 3), "weight_conv1")
    weight_conv2 = hcl.placeholder((64, 64, 3, 3), "weight_conv2")
    weight_conv3 = hcl.placeholder((128, 64, 3, 3), "weight_conv3")
    weight_conv4 = hcl.placeholder((128, 128, 3, 3), "weight_conv4")
    weight_conv5 = hcl.placeholder((256, 128, 3, 3), "weight_conv5")
    weight_conv6 = hcl.placeholder((256, 256, 3, 3), "weight_conv6")
    weight_conv7 = hcl.placeholder((256, 256, 3, 3), "weight_conv7")
    weight_conv8 = hcl.placeholder((512, 256, 3, 3), "weight_conv8")
    weight_conv9 = hcl.placeholder((512, 512, 3, 3), "weight_conv9")
    weight_conv10 = hcl.placeholder((512, 512, 3, 3), "weight_conv10")
    weight_conv11 = hcl.placeholder((512, 512, 3, 3), "weight_conv11")
    weight_conv12 = hcl.placeholder((512, 512, 3, 3), "weight_conv12")
    weight_conv13 = hcl.placeholder((512, 512, 3, 3), "weight_conv13")
    # weight_conv1 = hcl.placeholder((64, 3, 3, 3), "weight_conv1", qtype1)
    # weight_conv2 = hcl.placeholder((128, 64, 3, 3), "weight_conv2", qtype1)
    # weight_conv3 = hcl.placeholder((256, 256, 3, 3), "weight_conv3", qtype1)
    # weight_conv4 = hcl.placeholder((512, 256, 3, 3), "weight_conv4", qtype1)
    # weight_conv5 = hcl.placeholder((512, 512, 3, 3), "weight_conv5", qtype1)
    # weight_conv6 = hcl.placeholder((512, 512, 3, 3), "weight_conv6", qtype1)
    # weight_conv7 = hcl.placeholder((512, 512, 3, 3), "weight_conv7", qtype1)
    # weight_conv8 = hcl.placeholder((64, 3, 3, 3), "weight_conv1", qtype1)
    # weight_conv9 = hcl.placeholder((128, 64, 3, 3), "weight_conv2", qtype1)
    # weight_conv10 = hcl.placeholder((256, 256, 3, 3), "weight_conv3", qtype1)
    # weight_conv11 = hcl.placeholder((512, 256, 3, 3), "weight_conv4", qtype1)
    # weight_conv12 = hcl.placeholder((512, 512, 3, 3), "weight_conv5", qtype1)
    # weight_conv13 = hcl.placeholder((512, 512, 3, 3), "weight_conv6", qtype1)
    # weight_fc1 = hcl.placeholder((25088, 4096), "weight_fc1", qtype1)
    # weight_fc2 = hcl.placeholder((4096, 4096), "weight_fc2", qtype1)
    # weight_fc3 = hcl.placeholder((4096, 1000), "weight_fc3", qtype1)
    weight_fc1 = hcl.placeholder((4096, 25088), "weight_fc1")
    weight_fc2 = hcl.placeholder((4096, 4096), "weight_fc2")
    weight_fc3 = hcl.placeholder((1000, 4096), "weight_fc3")
    vgg = hcl.placeholder((batch_size, 1000), 'vgg')
    # create a quantization scheme
    scheme = hcl.create_scheme(
            [input_image, weight_conv1, weight_conv2, weight_conv3,
            weight_conv4, weight_conv5, weight_conv6, weight_conv7,
            weight_conv8, weight_conv9, weight_conv10, weight_conv11,
            weight_conv12, weight_conv13, weight_fc1, weight_fc2,
            weight_fc3, vgg], build_model)
    s = hcl.create_schedule_from_scheme(scheme)

    return hcl.build(s, target=target)

#f = build_model_inf()

###############################################################################
# Prepare the numpy arrays for testing. Remember that we need to set the input
# tensors with the same type as the placeholders
import mxnet as mx
sym, arg_params, aux_params = mx.model.load_checkpoint('vgg16', 0)
# print(arg_params.keys())
weight_conv1_np = arg_params['conv1_1_weight'].asnumpy()
weight_conv2_np = arg_params['conv1_2_weight'].asnumpy()
weight_conv3_np = arg_params['conv2_1_weight'].asnumpy()
weight_conv4_np = arg_params['conv2_2_weight'].asnumpy()
weight_conv5_np = arg_params['conv3_1_weight'].asnumpy()
weight_conv6_np = arg_params['conv3_2_weight'].asnumpy()
weight_conv7_np = arg_params['conv3_3_weight'].asnumpy()
weight_conv8_np = arg_params['conv4_1_weight'].asnumpy()
weight_conv9_np = arg_params['conv4_2_weight'].asnumpy()
weight_conv10_np = arg_params['conv4_3_weight'].asnumpy()
weight_conv11_np = arg_params['conv5_1_weight'].asnumpy()
weight_conv12_np = arg_params['conv5_2_weight'].asnumpy()
weight_conv13_np = arg_params['conv5_3_weight'].asnumpy()
weight_fc1_np = arg_params['fc6_weight'].asnumpy()
weight_fc2_np = arg_params['fc7_weight'].asnumpy()
weight_fc3_np = arg_params['fc8_weight'].asnumpy()
# dict_keys(['conv3_2_weight', 'fc7_bias', 'conv4_1_bias', 'conv5_3_bias', 'conv3_3_bias',
# 'conv4_3_weight', 'conv1_2_bias', 'conv4_1_weight', 'conv2_2_weight', 'fc7_weight', 'conv2_1_bias',
# 'conv5_2_weight', 'conv1_1_bias', 'conv4_3_bias', 'fc6_bias', 'conv5_3_weight', 'conv1_2_weight',
# 'fc8_bias', 'conv4_2_weight', 'conv3_1_weight', 'conv5_1_bias', 'conv4_2_bias', 'fc8_weight',
# 'conv2_1_weight', 'conv2_2_bias', 'conv5_1_weight', 'conv1_1_weight', 'conv3_1_bias', 'conv3_3_weight',
# 'fc6_weight', 'conv3_2_bias', 'conv5_2_bias'])

# weight_conv1_hcl = hcl.asarray(weight_conv1_np, dtype=qtype1)
# weight_conv2_hcl = hcl.asarray(weight_conv2_np, dtype=qtype1)
# weight_conv3_hcl = hcl.asarray(weight_conv3_np, dtype=qtype1)
# weight_conv4_hcl = hcl.asarray(weight_conv4_np, dtype=qtype1)
# weight_conv5_hcl = hcl.asarray(weight_conv5_np, dtype=qtype1)
# weight_conv6_hcl = hcl.asarray(weight_conv6_np, dtype=qtype1)
# weight_conv7_hcl = hcl.asarray(weight_conv7_np, dtype=qtype1)
# weight_fc1_hcl = hcl.asarray(weight_fc1_np, dtype=qtype1)
# weight_fc2_hcl = hcl.asarray(weight_fc2_np, dtype=qtype1)
# weight_fc3_hcl = hcl.asarray(weight_fc3_np, dtype=qtype1)
weight_conv1_hcl = hcl.asarray(weight_conv1_np)
weight_conv2_hcl = hcl.asarray(weight_conv2_np)
weight_conv3_hcl = hcl.asarray(weight_conv3_np)
weight_conv4_hcl = hcl.asarray(weight_conv4_np)
weight_conv5_hcl = hcl.asarray(weight_conv5_np)
weight_conv6_hcl = hcl.asarray(weight_conv6_np)
weight_conv7_hcl = hcl.asarray(weight_conv7_np)
weight_conv8_hcl = hcl.asarray(weight_conv8_np)
weight_conv9_hcl = hcl.asarray(weight_conv9_np)
weight_conv10_hcl = hcl.asarray(weight_conv10_np)
weight_conv11_hcl = hcl.asarray(weight_conv11_np)
weight_conv12_hcl = hcl.asarray(weight_conv12_np)
weight_conv13_hcl = hcl.asarray(weight_conv13_np)
weight_fc1_hcl = hcl.asarray(weight_fc1_np)
weight_fc2_hcl = hcl.asarray(weight_fc2_np)
weight_fc3_hcl = hcl.asarray(weight_fc3_np)

f = build_model_inf()

# im1 = Image.open("0.jpeg")
# im2 = im1.resize((224, 224))
# input_image_np = np.array(im2)
input_image_np = np.random.randn(batch_size, 3, 224, 224)

input_image_hcl = hcl.asarray(input_image_np)
output_hcl = hcl.asarray(np.zeros((batch_size, 1000)))
f(input_image_hcl, weight_conv1_hcl, weight_conv2_hcl, weight_conv3_hcl, weight_conv4_hcl,
weight_conv5_hcl, weight_conv6_hcl, weight_conv7_hcl, weight_conv8_hcl, weight_conv9_hcl,
weight_conv10_hcl, weight_conv11_hcl, weight_conv12_hcl, weight_conv13_hcl, weight_fc1_hcl,
weight_fc2_hcl, weight_fc3_hcl, output_hcl)

print(output_hcl)

