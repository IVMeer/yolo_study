from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='person.yaml', epochs=3)

model.val()