import taichi as ti
import copy
ti.init(arch=ti.opengl)
def _test_ndarray_deepcopy():
    n = 16
    x = ti.ndarray(ti.i32, shape=n)
    x[0] = 1
    x[4] = 2

    import pdb
    pdb.set_trace()
    y = copy.deepcopy(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y[0] == 1
    assert y[4] == 2

_test_ndarray_deepcopy()
