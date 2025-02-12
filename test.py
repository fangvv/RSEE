import torch.nn as nn
import torch
from thop import profile
import torchvision.models as models

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.LSTMCell(2048, 512)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 2048)
    
    def forward(self, x, h, c):
        # h = self.fc1(h)
        # print(x.shape, h.shape, c.shape)
        o1, c = self.l1(x, (h, c))
        o3 = self.fc2(o1)
        return o3, c

import time
model1 = models.alexnet()
print(model1)
x1 = torch.randn((1, 3, 84, 84))
for i in range(100):
    t1 = time.time()
    _ = model1(x1)
    print(time.time() - t1)
# model = Model()
# x = torch.randn((1, 2048))
# h, c = torch.randn((1, 512)), torch.randn((1, 512))
# flops, params = profile(model, (x, h, c))
# print(flops / 1e9, params / 1e6)
flops, params = profile(model1, x1)
print(flops / 1e9, params / 1e6)