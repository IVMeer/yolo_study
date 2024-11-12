from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='safehat.yaml', epochs=30)

# 使用验证集验证效果
model.val()