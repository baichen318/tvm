from tvm.tir import stmt as _stmt, expr as _expr
from tvm.tir import ir_pass as _pass
from tvm.tir import IterVar as _IterVar
from tvm.tir.expr import Var as _Var
from tvm.tir.ir_builder import WithScope
from .api import placeholder
from .debug import DSLError, APIError
from .schedule import Stage
from .module import Module
from . import util
from tvm.tir import _ffi_api

def and_(*args):
    """Compute the logic AND between expressions.

    If there is only one argument, itself is returned.

    Parameters
    ----------
    args : list of Expr
        A list of expression to be computed

    Returns
    -------
    Expr

    Examples
    --------
    .. code-block:: python

        A = hcl.placeholder((3,))
        cond = hcl.and_(A[0] > 0, A[1] > 1, A[2] > 2)
    """
    ret = args[0]
    for i in range(1, len(args)):
        ret = _expr.And(ret, args[i])
    return ret

def or_(*args):
    """Compute the logic OR between expressions.

    If there is only one argument, itself is returned.

    Parameters
    ----------
    args : list of Expr
        A list of expression to be computed

    Returns
    -------
    Expr

    Examples
    --------
    .. code-block:: python

        A = hcl.placeholder((3,))
        cond = hcl.or_(A[0] > 0, A[1] > 1, A[2] > 2)
    """
    ret = args[0]
    for i in range(1, len(args)):
        ret = _expr.Or(ret, args[i])
    return ret

def if_(cond):
    """Construct an IF branch.

    The usage is the same as Python `if` statement. Namely, a single `if`
    statement without the `else` branch is allowed. In addition, we cannot
    use `else` and `elif` without an `if` statement. Finally, an `else`
    statement must be preceded by either an `if` or `elif` statement.

    Parameters
    ----------
    cond : Expr
        The condition of the `if` statement

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        def my_compute(x):
            with hcl.if_(A[x] < 3):
                # do something
            with hcl.elif_(A[x] < 6):
                # do something
            with hcl.else_():
                # do something
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    stage = Stage.get_current()
    stage.stmt_stack.append([])
    def _exit_cb():
        stmt = stage.pop_stmt()
        stage.has_break = False
        stage.emit(_stmt.IfThenElse(cond, stmt, None))
    return WithScope(None, _exit_cb)

def else_():
    """Construct an ELSE branch.

    Parameters
    ----------

    Returns
    -------
    None

    See Also
    --------
    if_
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    stage = Stage.get_current()
    prev = stage.stmt_stack[-1][-1]
    if not isinstance(prev, _stmt.IfThenElse):
        raise DSLError("There is no if_ or elif_ in front of the else_ branch")
    stage.stmt_stack[-1].pop()
    stage.stmt_stack.append([])
    def _exit_cb():
        stmt = stage.pop_stmt()
        stage.has_break = False
        stage.emit(stage.replace_else(prev, stmt))
    return WithScope(None, _exit_cb)

def elif_(cond):
    """Construct an ELIF branch.

    Parameters
    ----------
    cond : Expr
        The condition of the branch

    Returns
    -------
    None

    See Also
    --------
    if_
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    stage = Stage.get_current()
    prev = stage.stmt_stack[-1][-1]
    if not isinstance(prev, _stmt.IfThenElse):
        raise DSLError("There is no if_ or elif_ in front of the elif_ branch")
    stage.stmt_stack[-1].pop()
    stage.stmt_stack.append([])
    def _exit_cb():
        stmt = stage.pop_stmt()
        stage.has_break = False
        stage.emit(stage.replace_else(prev, _stmt.IfThenElse(cond, stmt, None)))
    return WithScope(None, _exit_cb)

def for_(begin, end, step=1, name="i", dtype="int32", for_type="serial"):
    """Construct a FOR loop.

    Create an imperative for loop based on the given bound and step. It is
    the same as the following Python code.

    .. code-block:: python

        for i in range(begin, end, step):
            # do something

    The bound and step can be negative values. In addition, `begin` is
    inclusive while `end` is exclusive.

    Parameters
    ----------
    begin : Expr
        The starting bound of the loop

    end : Expr
        The ending bound of the loop

    step : Expr, optional
        The step of the loop

    name : str, optional
        The name of the iteration variable

    dtype : Type, optional
        The data type of the iteration variable

    for_type : str, optional
        The type of the for loop

    Returns
    -------
    Var
        The iteration variable

    See Also
    --------
    break_

    Examples
    --------
    .. code-block:: python

        # example 1 - basic usage
        with hcl.for_(0, 5) as i:
            # i = [0, 1, 2, 3, 4]

        # example 2 - negative step
        with hcl.for_(5, 0, -1) as i:
            # i = [5, 4, 3, 2, 1]

        # example 3 - larger step
        with hcl.for_(0, 5, 2) as i:
            # i = [0, 2, 4]

        # example 4 - arbitrary bound
        with hcl.for_(-4, -8, -2) as i:
            # i = [-4, -6]
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    stage = Stage.get_current()
    stage.stmt_stack.append([])
    extent = (end - begin) // step
    extent = util.CastRemover().mutate(extent)
    name = "i"+str(stage.for_ID) if name is None else name
    stage.for_ID += 1
    iter_var = _IterVar(_ffi_api.range_by_min_extent(0, extent), _Var(name, dtype), 0, '')
    stage.var_dict[name] = iter_var
    stage.axis_list.append(iter_var)
    stage.for_level += 1
    def _exit_cb():
        if for_type == "serial":
            for_type_id = 0
        elif for_type == "parallel":
            for_type_id = 1
        elif for_type == "vectorize":
            for_type_id = 2
        elif for_type == "unroll":
            for_type_id = 3
        else:
            raise ValueError("Unknown for_type")
        stmt = _stmt.AttrStmt(iter_var, "loop_scope", iter_var.var, stage.pop_stmt())
        stage.has_break = False
        stage.for_level -= 1
        stage.emit(_stmt.For(iter_var.var, 0, extent, for_type_id, 0, stmt))
    ret_var = _pass.Simplify(iter_var.var * step + begin)
    return WithScope(ret_var, _exit_cb)

def while_(cond):
    """Construct a WHILE loop.

    Parameters
    ----------
    cond : Expr
        The condition of the loop

    Returns
    -------
    None

    See Also
    --------
    break_

    Examples
    --------
    .. code-block:: python

        with hcl.while_(A[x] > 5):
            # do something
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    stage = Stage.get_current()
    stage.stmt_stack.append([])
    stage.for_level += 1
    def _exit_cb():
        stmt = stage.pop_stmt()
        stage.has_break = False
        stage.for_level -= 1
        stage.emit(_stmt.While(cond, stmt))
    return WithScope(None, _exit_cb)

def break_():
    """
    Construct a BREAK statement.

    This DSL can only be used inside a `while` loop or a `for loop`. Moreover,
    it is not allowed to have tracing statements after the `break`.

    Parameters
    ----------

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        # example 1 - inside a for loop
        with hcl.for_(0, 5) as i:
            with hcl.if_(A[i] > 5):
                hcl.break_()

        # example 2 - inside a while loop
        with hcl.while_(A[i] > 5):
            with hcl.if_(A[i] > 10):
                hcl.break_()
    """
    if not Stage.get_len():
        raise DSLError("Imperative DSL must be used with other compute APIs")
    if not Stage.get_current().for_level:
        raise DSLError("break_ must be used inside a for/while loop")
    Stage.get_current().emit(_stmt.Break())
    Stage.get_current().has_break = True