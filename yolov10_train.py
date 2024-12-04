from ultralytics import YOLO
# Load a pretrained yolov8n model
model = YOLO('yolov10n.pt')

# Train the model using the 'safehat.yaml' dataset for 10 epochs
model.train(data='safehat.yaml', epochs=10)

# 使用验证集验证效果
model.val()