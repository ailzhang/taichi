import numpy as np
import pytest
from taichi.lang.exception import TaichiCompilationError

import taichi as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan)
def test_ndarray_int():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(field_dim=1)):
        for i in range(n):
            pos[i] = 1

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                           'pos',
                           ti.i32,
                           field_dim=1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.i32, shape=(n, ))
    g.run({'pos': a})
    assert (a.to_numpy() == np.ones(4)).all()


@test_utils.test(arch=ti.vulkan)
def test_ndarray_float():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(field_dim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                           'pos',
                           ti.f32,
                           field_dim=1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.f32, shape=(n, ))
    g.run({'pos': a})
    assert (a.to_numpy() == (np.ones(4) * 2.5)).all()


@test_utils.test(arch=ti.vulkan)
def test_arg_mismatched_field_dim():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(field_dim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                           'pos',
                           ti.f32,
                           field_dim=2)
    g_init = ti.graph.GraphBuilder()
    with pytest.raises(TaichiCompilationError,
                       match="doesn't match kernel's annotated field_dim"):
        g_init.dispatch(test, sym_pos)


@test_utils.test(arch=ti.vulkan)
def test_arg_mismatched_field_dim_ndarray():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(field_dim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'pos', ti.f32, 1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.f32, shape=(n, n))
    with pytest.raises(RuntimeError, match="Dispatch node is compiled for"):
        g.run({'pos': a})


@test_utils.test(arch=ti.vulkan)
def test_repeated_arg_name():
    n = 4

    @ti.kernel
    def test1(pos: ti.types.ndarray(field_dim=1)):
        for i in range(n):
            pos[i] = 2.5

    @ti.kernel
    def test2(v: ti.f32):
        for i in range(n):
            print(v)

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                           'pos',
                           ti.f32,
                           field_dim=1)
    sym_pos1 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'pos', ti.f32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(test1, sym_pos)

    with pytest.raises(RuntimeError):
        builder.dispatch(test2, sym_pos1)


def build_graph_vector(N, dtype):
    @ti.kernel
    def vector_sum(mat: ti.types.vector(N, dtype),
                   res: ti.types.ndarray(field_dim=1)):
        res[0] = mat.sum() + mat[2]

    sym_A = ti.graph.Arg(ti.graph.ArgKind.MATRIX, 'mat',
                         ti.types.vector(N, dtype))
    sym_res = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'res', dtype, field_dim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(vector_sum, sym_A, sym_res)
    graph = builder.compile()
    return graph


def build_graph_matrix(N, dtype):
    @ti.kernel
    def matrix_sum(mat: ti.types.matrix(N, 2, dtype),
                   res: ti.types.ndarray(field_dim=1)):
        res[0] = mat.sum()

    sym_A = ti.graph.Arg(ti.graph.ArgKind.MATRIX, 'mat',
                         ti.types.matrix(N, 2, dtype))
    sym_res = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'res', dtype, field_dim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(matrix_sum, sym_A, sym_res)
    graph = builder.compile()
    return graph


@test_utils.test(arch=ti.vulkan)
def test_matrix_int():
    n = 4
    A = ti.Matrix([4, 5] * n)
    res = ti.ndarray(ti.i32, shape=(1, ))
    graph = build_graph_matrix(n, dtype=ti.i32)
    graph.run({"mat": A, "res": res})
    assert (res.to_numpy()[0] == 36)


@test_utils.test(arch=ti.vulkan)
def test_matrix_float():
    n = 4
    A = ti.Matrix([4.2, 5.7] * n)
    res = ti.ndarray(ti.f32, shape=(1))
    graph = build_graph_matrix(n, dtype=ti.f32)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == test_utils.approx(39.6, rel=1e-5)


@test_utils.test(arch=ti.vulkan)
def test_vector_int():
    n = 12
    A = ti.Vector([1, 3, 13, 4, 5, 6, 7, 2, 3, 4, 1, 25])
    res = ti.ndarray(ti.i32, shape=(1, ))
    graph = build_graph_vector(n, dtype=ti.i32)
    graph.run({"mat": A, "res": res})
    assert (res.to_numpy()[0] == 87)


@test_utils.test(arch=ti.vulkan)
def test_vector_float():
    n = 8
    A = ti.Vector([1.4, 3.7, 13.2, 4.5, 5.6, 6.1, 7.2, 2.6])
    res = ti.ndarray(ti.f32, shape=(1, ))
    graph = build_graph_vector(n, dtype=ti.f32)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == test_utils.approx(57.5, rel=1e-5)
