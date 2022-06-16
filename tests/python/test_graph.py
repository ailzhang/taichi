import numpy as np
import pytest

import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan)
def test_ndarray_int():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(field_dim=1)):
        for i in range(n):
            pos[i] = 1

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'pos', ti.i32)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.i32, shape=(n, ))
    g.run({'pos': a})
    assert (a.to_numpy() == np.ones(4)).all()


@pytest.mark.parametrize('dt', [ti.f32, ti.f64])
@test_utils.test(arch=ti.vulkan)
def test(dt):
    @ti.kernel
    def foo(a: float, res: ti.types.ndarray(field_dim=1)):
        res[0] = a

    x = 122.33

    res_arr = ti.ndarray(dt, shape=(1, ))

    sym_A = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'a', dt)
    sym_res = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'res', dt)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(foo, sym_A, sym_res)
    graph = builder.compile()
    graph.run({"a": x, "res": res_arr})
    assert res_arr[0] == x
