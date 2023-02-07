import taichi as ti
ti.init(ti.vulkan, print_ir=False)

@ti.kernel
def test(x: ti.f32) -> ti.f32:
    a, b = ti.frexp(x)
    return b


print(test(1.4))
