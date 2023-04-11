import numpy as np
import pytest

import taichi as ti

ti.init(ti.cpu, print_ir=True)


def test_struct_arg():
    s0 = ti.types.struct(a=ti.i16, b=ti.f64)
    x = ti.ndarray(dtype=ti.f32, shape=(4))
    x[0] = 2

    #@ti.kernel
    #def foo(a: s0, x: ti.types.ndarray()) -> ti.f32:
    #    return a.a + a.b + x[0]

    #ret = foo(s0(a=1, b=123), x)
    #assert ret == pytest.approx(125)
    @ti.kernel
    def foo(x: ti.types.ndarray()) -> ti.f32:
        return x[0]

    ret = foo(x)
    print(ret)


test_struct_arg()
