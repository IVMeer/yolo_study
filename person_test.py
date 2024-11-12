from ultralytics import YOLO 

model = YOLO('E:\\workspace\\yolo_demo\\runs\\detect\\train3\\weights\\best.pt')

# test


model.predict('E:\\workspace\\yolo_data\\person-data\\test\\1.jpg')
model.predict('E:\\workspace\\yolo_data\\person-data\\test\\2.jpg')
model.predict('E:\\workspace\\yolo_data\\person-data\\video\\1.mp4', save = True, classes=[5], line_width=2)
# model.predict('E:\\workspace\\yolo_data\\person-data\\video\\2.mp4', save = True, classes=[5], line_width=30)