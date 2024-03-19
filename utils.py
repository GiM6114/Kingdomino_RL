import numpy as np

def arr_except(arr, except_id):
    return np.concatenate((arr[:except_id],arr[except_id+1:]))

def switch(l, i, j):
    lc = l.copy()
    lc[i],lc[j] = lc[j],lc[i]
    return lc

# https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)