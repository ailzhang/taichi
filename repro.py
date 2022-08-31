import torch

import taichi as ti

ti.init(arch=ti.gpu, log_level=ti.TRACE)
print("===AFTER TI INIT===")
# print_existing_contexts()
device = torch.device("cuda:0")
print("===AFTER TORCH DEVICE INIT===")
# print_existing_contexts()
print(torch._C._cuda_hasPrimaryContext(0))
x = torch.tensor([1.], requires_grad=True, device=device)
print("===AFTER TORCH TENSOR INIT===")
# print_existing_contexts()
# print("TAICHI "ti._lib.core.get_primary_ctx_state())
print("Torch has primary context", torch._C._cuda_hasPrimaryContext(0))
loss = x**2
loss.backward()
print(torch._C._cuda_hasPrimaryContext(0))
