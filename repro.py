import taichi as ti

ti.init(print_ir=True)

a = ti.field(ti.f32, shape=(4), needs_grad=True)
b = ti.field(ti.f32, shape=(), needs_grad=True)


@ti.kernel
def test():
    for i in a:
        b[None] += a[i] * 3

    for i in a:
        b[None] += a[i] * 5


a.fill(1)
with ti.ad.Tape(loss=b):
    test()

print(b)
print(a.grad)
