import torch
import cupy

cupy.show_config()
print(torch.backends.cudnn)
print(torch.version.cuda)