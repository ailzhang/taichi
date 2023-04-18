import torch

import taichi as ti

ti.init(print_ir=False)


@ti.kernel
def test(x: ti.types.ndarray(), y: ti.types.ndarray()):
    for i in x:
        a = 1.0
        for j in range(4):
            a += x[i] / 3
        y[0] += a


# x = torch.ones((4, ), requires_grad=True)
# y = torch.ones((1, ), requires_grad=True)
x = ti.ndarray(ti.f32, shape=(4, ), needs_grad=True)
x.fill(1.0)
y = ti.ndarray(ti.f32, shape=(1, ), needs_grad=True)
y.fill(1.0)
with ti.ad.Tape(loss=y):
    test(x, y)
# test(x, y)
print(y.to_numpy())
# print(y)

print(x.grad.to_numpy())
