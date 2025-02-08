import torch

from torch import nn
from torchvision.transforms.functional import crop

a = torch.randn(1,2,10,10)


m = nn.ConvTranspose2d(2, 1, 2, stride = 2)


print(a.size())

b = m(a)

print(b.size())

c = crop(a, 2,2,6,6)

print(c.size())


print(c)

