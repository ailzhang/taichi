import torch

import taichi as ti

ti.init(print_ir=True)


@ti.kernel
def test(x: ti.types.ndarray(), y: ti.types.ndarray()):
    for i in x:
        a = 1.0
        for j in range(4):
            a += x[i] / 3
        y[0] += a


x = torch.ones((4, ), requires_grad=True)
y = torch.ones((1, ), requires_grad=True)

with ti.ad.Tape(loss=y):
    test(x, y)

print(y)

print(x.grad)
