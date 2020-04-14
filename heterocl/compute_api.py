"""Compute APIs in HeteroCL"""
#pylint: disable=no-member, redefined-builtin, too-many-arguments, missing-docstring
import numbers
from collections import OrderedDict
from tvm import expr_hcl as _expr, stmt as _stmt
from tvm.tir import IterVar as _IterVar
from util import get_index, get_name, make_for, CastRemover
from tensor import Scalar, Tensor, TensorSlice
from schedule import Stage
from debug import APIError
from module import Module
##############################################################################
# Helper classes and functions
##############################################################################

class ReplaceReturn(CastRemover):
    """Replace all Return statement with a Store statement.

    Attributes
    ----------
    buffer_var : Var
        The buffer variable of the Store statement

    dtype : Type
        The data type of the Store statement

    index : Expr
        The index of the Store statement
    """
    def __init__(self, buffer_var, dtype, index):
        self.buffer_var = buffer_var
        self.dtype = dtype
        self.index = index

    def mutate_KerenlDef(self, node):
        """Omit the KernelDef statement

        We do not need to replace the Return statement inside.
        """
        #pylint: disable=no-self-use
        return node

    def mutate_Return(self, node):
        """Replace the Return statement with a Store statement

        """
        return _make.Store(self.buffer_var, _make.Cast(self.dtype, node.value), self.index)

def process_fcompute(fcompute, shape):
    """Pre-process the fcompute field of an API.

    """
    # check API correctness
    if not callable(fcompute):
        raise APIError("The construction rule must be callable")
    # prepare the iteration variables
    args = [] # list of arguments' names
    nargs = 0 # number of arguments
    if isinstance(fcompute, Module):
        args = fcompute.arg_names
        nargs = len(args)
    else:
        args = list(fcompute.__code__.co_varnames)
        nargs = fcompute.__code__.co_argcount
    # automatically create argument names
    if nargs < len(shape):
        for i in range(nargs, len(shape)):
            args.append("args" + str(i))
    elif nargs > len(shape):
        raise APIError("The number of arguments exceeds the number of dimensions")
    return args, len(shape)

def compute_body(name,
                lambda_ivs,
                fcompute,
                shape=(),
                dtype=None,
                tensor=None,
                attrs=OrderedDict()):
    """Create a stage and perform the computation.

    If `tensor` is `None`, no tensor is returned.

    Parameters
    ----------
    name : str
        The name of the stage

    lambda_ivs : list of IterVar
        A list contains the iteration variables in the lambda function if
        exists

    fcompute : callable
        The computation rule

    shape : tuple, optional
        The output shape or the iteration domain

    dtype : Type, optional
        The data type of the output/updated tensor

    tensor : Tensor, optional
        The tensor to be updated. Create a new one if it is `None`

    Returns
    -------
    Tensor or None
    """
    var_list = [i.var for i in lambda_ivs]
    return_tensor = True if tensor is None else False

    with Stage(name, dtype, shape) as stage:
        if not return_tensor:
            stage.input_stages.add(tensor.last_update)
        else:
            tensor = Tensor(shape, stage._dtype, name, stage._buf)
        buffer_var = tensor._buf.data
        dtype = tensor.dtype
        shape = tensor.shape

        stage.stmt_stack.append([])
        ret = fcompute(*var_list)

        print(dir(ret))
        print(dir(ret.a))
        print(dir(ret.b))
        stage.lhs_tensors.add(tensor)
        for t in stage.lhs_tensors:
            t.last_update = stage

        stmt = None
        if ret is None:
            # replace all hcl.return_ with Store stmt
            indices = lambda_ivs
            index, _, _ = get_index(shape, indices, 0)
            stmt = stage.pop_stmt()
            stmt = ReplaceReturn(buffer_var, dtype, index).mutate(stmt)
            stmt = make_for(indices, stmt, 0)
        elif isinstance(ret, (TensorSlice, Scalar, _expr.Expr, numbers.Number)):
            indices = lambda_ivs
            index, _, _ = get_index(shape, indices, 0)
            stage.emit(_make.Store(buffer_var, _make.Cast(dtype, ret), index))
            stmt = make_for(indices, stage.pop_stmt(), 0)
        elif isinstance(ret, Tensor): # reduction
            ret_ivs = [_IterVar((0, ret.shape[i]), ret.name+"_i" + str(i), 0)
                       for i in range(0, len(ret.shape))]
            non_reduce_ivs = []
            indices = []
            rid = 0
            for iv in lambda_ivs:
                if iv.var.name[0] == "_":
                    indices.append(ret_ivs[rid])
                    rid += 1
                else:
                    indices.append(iv)
                    non_reduce_ivs.append(iv)
            if rid != len(ret.shape):
                raise APIError("Incorrect number of reduction axes in lambda arguments")
            index, _, _ = get_index(shape, indices, 0)
            st = _make.Store(buffer_var, _make.Cast(dtype, ret[tuple(ret_ivs)]), index)
            stage.emit(make_for(ret_ivs, st, 0))
            stmt = stage.pop_stmt()
            stage.input_stages.remove(stage)
            if non_reduce_ivs:
                stmt = make_for(non_reduce_ivs, stmt, 0)
        else:
            raise APIError("Unknown return type of the computation rule")
        # add attributes to the loop
        if isinstance(stmt, _stmt.For):
            stmt = _make.For(stmt.loop_var,
                             stmt.min, stmt.extent,
                             0, 0, stmt.body,
                             list(attrs.keys()),
                             list(attrs.values()))
        stage.emit(stmt)
        stage.axis_list = indices + stage.axis_list

    if return_tensor:
        tensor._tensor = stage._op
        return tensor
    return None

##############################################################################
# APIs exposed to users
##############################################################################

def compute(shape, fcompute, name=None, dtype=None, attrs=OrderedDict()):
    """Construct a new tensor based on the shape and the compute function.

    The API **returns a new tensor**. The shape must be a tuple. The number of
    elements in the tuple decides the dimension of the returned tensor. The
    second field `fcompute` defines the construction rule of the returned
    tensor, which must be callable. The number of arguments should match the
    dimension defined by `shape`, which *we do not check*. This, however,
    provides users more programming flexibility.

    The compute function specifies how we calculate each element of the
    returned tensor. It can contain other HeteroCL APIs, even imperative DSL.

    Parameters
    ----------
    shape : tuple
        The shape of the returned tensor

    fcompute : callable
        The construction rule for the returned tensor

    name : str, optional
        The name of the returned tensor

    dtype : Type, optional
        The data type of the placeholder

    Returns
    -------
    Tensor

    Examples
    --------
    .. code-block:: python

        # example 1.1 - anonymous lambda function
        A = hcl.compute((10, 10), lambda x, y: x+y)

        # equivalent code
        for x in range(0, 10):
            for y in range(0, 10):
                A[x][y] = x + y

        # example 1.2 - explicit function
        def addition(x, y):
            return x+y
        A = hcl.compute((10, 10), addition)

        # example 1.3 - imperative function definition
        @hcl.def_([(), ()])
        def addition(x, y):
            hcl.return_(x+y)
        A = hcl.compute((10, 10), addition)

        # example 2 - undetermined arguments
        def compute_tanh(X):
            return hcl.compute(X.shape, lambda *args: hcl.tanh(X[args]))

        A = hcl.placeholder((10, 10))
        B = hcl.placeholder((10, 10, 10))
        tA = compute_tanh(A)
        tB = compute_tanh(B)

        # example 3 - mixed-paradigm programming
        def return_max(x, y):
            with hcl.if_(x > y):
                hcl.return_(x)
            with hcl.else_:
                hcl.return_(y)
        A = hcl.compute((10, 10), return_max)
    """
    # check API correctness
    if not isinstance(shape, tuple):
        raise APIError("The shape of compute API must be a tuple")

    # properties for the returned tensor
    shape = CastRemover().mutate(shape)
    name = get_name("compute", name)

    # prepare the iteration variables
    args, nargs = process_fcompute(fcompute, shape)
    lambda_ivs = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]

    # call the helper function that returns a new tensor
    tensor = compute_body(name, lambda_ivs, fcompute, shape, dtype, attrs=attrs)

    return tensor