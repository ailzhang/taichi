import taichi as ti

ti.init(arch=ti.cuda,
        print_ir=True,
        print_kernel_llvm_ir=True,
        print_kernel_nvptx=True)
dtype = ti.f16
# dtype = ti.f32
x = ti.field(dtype, shape=())
z = ti.field(dtype, shape=())


@ti.kernel
def bar(y: float):
    x[None] = z[None] + 0.3
    print(x[None])


z[None] = 0.3
print(z[None])

bar(0.21)
