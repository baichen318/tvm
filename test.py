import os,sys
tempdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,tempdir+"/heterocl")

import heterocl as hcl
import numpy as np

hcl.init()

A = hcl.placeholder((10,))

def quantization(A):
    return hcl.compute(A.shape, lambda x: A[x] + 1, "B")

sm = hcl.create_scheme([A], quantization)
sm_B = quantization.B
#sm.quantize(sm_B, hcl.Fixed(10, 8))
sl = hcl.create_schedule_from_scheme(sm)
f = hcl.build(sl)

#hcl_A = hcl.asarray(np.random.rand(1)*2-1)
#hcl_BQ = hcl.asarray(np.zeros(1), dtype = hcl.Fixed(10, 8))

#f(hcl_A, hcl_BQ)

#np_A = hcl_A.asnumpy()
#np_BQ = hcl_BQ.asnumpy()

#print(np_A)
#print('Quantized to Fixed(10, 8)')
#print(np_BQ)