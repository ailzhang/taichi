import taichi as ti

ti.init(arch=ti.cuda, print_ir=False, print_kernel_llvm_ir=False)
dtype = ti.f16
# dtype = ti.f32
x = ti.field(dtype, shape=())
z = ti.field(dtype, shape=())


@ti.kernel
def bar(y: float):
    x[None] = y + 0.3
    print(x[None])


z[None] = 0.3
print(z[None])

bar(0.21)
