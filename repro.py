import taichi as ti
ti.init(ti.cuda)

@ti.kernel
def test(x: ti.f32) -> ti.f32:
    return ti.frexp(x)


print(test(16.4))
