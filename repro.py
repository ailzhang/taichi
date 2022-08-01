import taichi as ti

ti.init(arch=ti.opengl, log_level=ti.TRACE)

x = ti.field(dtype=ti.f32, shape=())


@ti.kernel
def mixed_inner_loops():
    for i in range(1):
        x[None] += 1
        # for j in range(1):
        # x[None] = 1


mixed_inner_loops()
