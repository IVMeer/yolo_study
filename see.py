# import torch

# # 加载模型
# model = torch.load('.\\runs\\detect\\train\\weights\\best.pt')

# print(model)
# import torch
# from torchsummary import summary


# model = torch.load('.\\runs\\detect\\train\\weights\\best.pt')

# summary(model, input_size=(3, 224,224))
import torch
from ultralytics import YOLO

checkpoint = torch.load('.\\runs\\detect\\train\\weights\\best.pt')

model = checkpoint['model']

print(model)