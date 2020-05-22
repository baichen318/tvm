from collections import OrderedDict
import heterocl as hcl
import tvm
import tvm.tir as tir
import tvm.te as te
from topi.util import equal_const_int

dtype = hcl.Float()

sum = hcl.reducer(0, lambda x, y: x + y, dtype)
max = hcl.reducer(-1, lambda x, y: tir.expr.Max(x, y), dtype)

def simplify(expr):
    return tvm.tir.ir_pass.Simplify(expr) if isinstance(expr, tir.expr.PrimExpr) else expr

def relu(x):
    return max(0, x)

def pad(data, pad_before, pad_after=None, pad_value=0.0, name="pad"):
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError(
            "Input dimension and pad_before dismatch : %d vs %d" %
            (n, len(pad_before)))
    if len(pad_after) != n:
        raise ValueError(
            "Input dimension and pad_after dismatch : %d vs %d" %
            (n, len(pad_after)))
    out_shape = tuple(
        tvm.tir.ir_pass.Simplify(
            (data.shape[i] + tir.const(pad_before[i]) + tir.const(pad_after[i]))) for i in range(n))
    def _pad(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
                index_tuple.append(indices[i])
            else:
                index_tuple.append(indices[i] - pad_before[i])
                not_zero.append(indices[i] >= pad_before[i])
                not_zero.append(indices[i] < data.shape[i] + pad_before[i])
        if not_zero:
            not_zero = tir.all(*not_zero)
            return tir.Select(not_zero, data[tuple(index_tuple)], pad_value)
        return data[tuple(index_tuple)]

    return hcl.compute(out_shape, _pad, name=name)

def conv2d_nchw(
        Input,
        Filter,
        name="conv2d",
        stride=[1, 1],
        padding=[[0, 0], [0, 0]]):
    out_dtype = Input.dtype
    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    stride_h, stride_w = stride
    [pad_top, pad_left], [pad_down, pad_right] = padding
    # compute the output shape
    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    if padding != [[0, 0], [0, 0]]:
        Input = pad(Input, pad_before, pad_after)
    rc = hcl.reduce_axis(0, in_channel)
    ry = hcl.reduce_axis(0, kernel_h)
    rx = hcl.reduce_axis(0, kernel_w)
    return hcl.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: sum(
            Input[nn, rc, yy * stride_h + ry, xx * stride_w + rx] *
            Filter[ff, rc, ry, rx],
            axis=[rc, ry, rx]),
        name=name,
        attrs=OrderedDict([
            ('p', kernel_h),
            ('q', kernel_w),
            ('in_num', in_channel),
            ('out_num', out_channel),
            ('out_img_w', out_width),
            ('out_img_h', out_height),
            ('cin_dtype', tir.expr.StringImm(Input.dtype)),
            ('filter_dtype', tir.expr.StringImm(Filter.dtype)),
            ('app_name', tir.expr.StringImm('cnn'))]))


def bias_add(data, bias, axis=-1, name='bias_add'):
    def _broadcast(shape, *indices):
        axes = []
        indices = indices[0]
        for i in range(len(shape)):
            if shape[i] == 1:
                axes.append(0)
            else:
                axes.append(indices[i])
        return tuple(axes)
    data_len = len(data.shape)
    if axis < 0:
        axis += data_len
    num_newaxis = data_len - axis - 1
    bias = expand_dims(bias, len(bias.shape), num_newaxis)
    bias = expand_dims(bias, 0, axis)
    return hcl.compute(
        data.shape, lambda *x: data[x] + bias[_broadcast(bias.shape, x)], name=name)


def dense(data, weight, bias=None, name="dense"):
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = hcl.reduce_axis(0, in_dim)
    attrs=OrderedDict([
        ('k', in_dim),
        ('j', out_dim),
        ('i', batch),
        ('app_name', tir.expr.StringImm('mm'))])
    matmul = hcl.compute((batch, out_dim), lambda i, j: sum(data[i, k] * weight[j, k], axis=k), name, attrs=attrs)
    if bias is not None:
        matmul = hcl.compute(
                (batch, out_dim),
                lambda i, j: matmul[i, j] + bias[j],
                name=name,
                attrs=attrs)
    return matmul

def tanh(x, name="tanh"):
    return hcl.compute(x.shape, lambda *args: te.tanh(x[args]), name,
                       attrs=OrderedDict([('app_name', tir.expr.StringImm('tanh'))]))

def max_pool(data, kernel, stride, padding=[[0,0],[0,0]], name="max_pool"):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride
    batch, channel, height, width = data.shape
    [pad_top, pad_left], [pad_down, pad_right] = padding
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    if padding != [[0,0],[0,0]]:
        data = pad(data, pad_before, pad_after, pad_value=tir.min_value("float32"))
    out_height = simplify((height - kernel_height + pad_top + pad_down) // stride_height + 1)
    out_width = simplify((width - kernel_width + pad_left + pad_right) // stride_width + 1)
    dheight = hcl.reduce_axis(0, kernel_height)
    dwidth = hcl.reduce_axis(0, kernel_width)
    return hcl.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: max(data[i, c, h*stride_height+dheight, w*stride_width+dwidth], axis=[dheight, dwidth]),
        name=name,
        attrs=OrderedDict([
            ('out_img_w', out_width),
            ('out_img_h', out_height),
            ('in_num', channel),
            ('kernel_h', kernel[1]),
            ('kernel_w', kernel[0]),
            ('stride_h', stride[1]),
            ('stride_w', stride[0]),
            ('app_name', tir.expr.StringImm('max_pool'))]))

def flatten(data):
    ishape = data.shape
    dim = 1
    for i in range(1, len(ishape)):
        dim = dim * ishape[i]
    oshape = (ishape[0], dim)

    def unwrap(idx, shape):
        index = []
        for s in reversed(shape):
            index.append(idx % s)
            idx = te.div(idx, s)
        return list(reversed(index))

    return hcl.compute(oshape, lambda i, j: data[tuple([i] + unwrap(j, ishape[1:]))],
                       attrs=OrderedDict([('app_name', tir.expr.StringImm('flatten'))]))

def softmax(out, x):
    assert len(x.shape) == 2, "only support 2-dim softmax"
    m, n = x.shape
    k = hcl.reduce_axis(0, n)
    max_elem = hcl.compute((m, ), lambda i: hcl.hcl_max(x[i, k], axis=k))
    k = hcl.reduce_axis(0, n)
    expsum = hcl.compute(
        (m, ), lambda i: sum(hcl.exp(x[i, k] - max_elem[i]), axis=k))
    return hcl.update(
        out, lambda i, j: hcl.exp(x[i, j] - max_elem[i]) / expsum[i])

