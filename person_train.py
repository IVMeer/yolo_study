from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='person.yaml', epochs=30)

model.val()