import os

import torch

print(torch.__version__)
print(torch.cuda.is_available())

print(f"mps:{torch.backends.mps.is_available()}")  # the MacOS is higher than 12.3+
print(f"MPS:{torch.backends.mps.is_built()}")  # MPS is activated
print(f"openmp:{torch.backends.openmp.is_available()}")

mps_device = torch.device("mps")
x = torch.ones(1, device=mps_device)
print(x)  # 输出tensor([1.], device='mps:0')

result = torch.tensor(1) + torch.tensor(2)
print(result)
print("OMP_PREFIXOMP_PREFIXOMP_PREFIXOMP_PREFIXOMP_PREFIX", os.getenv("OMP_PREFIX"))
