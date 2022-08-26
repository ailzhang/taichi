import torch

import taichi as ti

ti.init(arch=ti.gpu, log_level=ti.TRACE)
print(ti._lib.core.get_primary_ctx_state())
# print(dir(ti._lib.core))
device = torch.device("cuda:0")
print(torch._C._cuda_hasPrimaryContext(0))

x = torch.tensor([1.], requires_grad=True, device=device)
print(ti._lib.core.get_primary_ctx_state())
print(torch._C._cuda_hasPrimaryContext(0))
loss = x**2
loss.backward()
print(torch._C._cuda_hasPrimaryContext(0))
