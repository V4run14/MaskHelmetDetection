import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
# conda install pytorch torchvision cudatoolkit=11.3 -c pytorch