import taichi as ti
import torch

ti.init(arch=ti.cuda, kernel_profiler=True, ad_stack_size=1024, print_ir=False, print_kernel_llvm_ir=False)

data_type = ti.f32
batch_size = 8192
# samples level
sigmas_fields = ti.field(dtype=data_type, shape=(batch_size * 1024,), needs_grad=True)
# rays level
rays_a_fields = ti.field(dtype=ti.i64, shape=(batch_size, 3))
rgb_fields = ti.field(dtype=data_type, shape=(batch_size, 3), needs_grad=True)

N_ITER = 2


@ti.kernel
def torch2ti(field: ti.template(), data: ti.types.ndarray()):
    for I in ti.grouped(data):
        field[I] = data[I]


@ti.kernel
def composite_train_fw(
    sigmas: ti.template(),
    rays_a: ti.template(),
    rgb: ti.template(),
):
    ti.loop_config(block_dim=256)
    for n in range(batch_size):
        ray_idx = ti.i32(rays_a[n, 0])
        # ray_idx = 0

        rgb[ray_idx, 0] = 0.0

        for sample_ in range(N_ITER):
            rgb[ray_idx, 0] += sigmas[sample_]


@ti.kernel
def composite_train_fw_stack(
    sigmas: ti.template(),
    rays_a: ti.template(),
    rgb: ti.template(),
):
    ti.loop_config(block_dim=256)
    for n in range(batch_size):
        ray_idx = ti.i32(rays_a[n, 0])
        # ray_idx = 0

        rgb_0 = 0.0

        # for sample_ in ti.static(range(N_ITER)):
        for sample_ in range(N_ITER):
            rgb_0 += sigmas[sample_]
        # rgb_0 += sigmas[0]
        # rgb_0 += sigmas[1]

        rgb[ray_idx, 0] = rgb_0


dirs = torch.load("./test_data/dir.t").float()
sigmas = torch.load("./test_data/sigmas.t").float()
rays_a = torch.load("./test_data/rays_a.t")


def run_without_stack():
    torch2ti(sigmas_fields, sigmas.contiguous())
    torch2ti(rays_a_fields, rays_a.contiguous())
    rgb_fields.fill(0.0)

    sigmas_fields.grad.fill(0)
    rgb_fields.grad.fill(0)

    rgb_fields.grad.fill(1.0)

    composite_train_fw(
        sigmas_fields,
        rays_a_fields,
        rgb_fields,
    )

    print(rgb_fields.to_numpy().sum())

    composite_train_fw.grad(
        sigmas_fields,
        rays_a_fields,
        rgb_fields,
    )

    print("no stack", sigmas_fields.grad.to_numpy().sum())


def run_with_stack():
    torch2ti(sigmas_fields, sigmas.contiguous())
    torch2ti(rays_a_fields, rays_a.contiguous())
    rgb_fields.fill(0.0)

    sigmas_fields.grad.fill(0)
    rgb_fields.grad.fill(0)

    rgb_fields.grad.fill(1.0)

    composite_train_fw_stack(
        sigmas_fields,
        rays_a_fields,
        rgb_fields,
    )
    print(rgb_fields.to_numpy().sum())

    composite_train_fw_stack.grad(
        sigmas_fields,
        rays_a_fields,
        rgb_fields,
    )

    print("with stack", sigmas_fields.grad.to_numpy().sum())


# Warmup
run_with_stack()
run_without_stack()


ti.profiler.clear_kernel_profiler_info()
run_with_stack()
ti.profiler.print_kernel_profiler_info()

ti.profiler.clear_kernel_profiler_info()
run_without_stack()
ti.profiler.print_kernel_profiler_info()
