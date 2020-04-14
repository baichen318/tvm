from tvm.tir import expr as _expr

a = _expr.IntImm("int32", 1)

print(type(a.dtype))
print(type(a.value))

class b():
    def __init__(self, value):
        self.value = value

res = b(123)

print(dir(b))