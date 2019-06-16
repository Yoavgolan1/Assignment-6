import backprop as bp
from srn import SRN
import numpy as np
import random

def xorseq(n):
    out_seq = np.array(np.zeros((n*3,1)))
    for i in range(n):
        out_seq[3*i, 0] = np.random.randint(2)
        out_seq[3*i+1, 0] = np.random.randint(2)
        out_seq[3*i+2, 0] = out_seq[3*i, 0] != out_seq[3*i+1, 0]
    return out_seq

def shift(x):
    out_arr = np.array(np.zeros((x.size, 1)))
    for i in range(x.size-1):
        out_arr[i, 0] = x[i+1, 0]
    return out_arr



backp = bp.Backprop(1, 1, 2)
inputs = xorseq(10)
#print(inputs)
targets = shift(inputs)
#backp.train(inputs, targets, 600, 0.5, 0, 0)


my_srn = SRN(1, 1, 2)
my_srn.train(inputs, targets, 600, 0.5, 0, 0)
print(my_srn.test(inputs))

#print(inputs)
#print(targets)
#my_srn = SRN(1, 2, 1)
#my_srn.train(inputs, targets, 0.5, 600)
