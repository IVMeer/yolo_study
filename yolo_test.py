from ultralytics import YOLO
# 导入训练好的模型best.py
model = YOLO('.\\runs\\detect\\train3\\weights\\best.pt')
# 随意找一些测试数据
# 图片数据和视频数都可以，直接将数据传入接口
model.predict('.\\test1.png', save=True)
model.predict('.\\test2.png', save=True)
model.predict('.\\test3.png', save=True)
# model.predict('xxx.mp4', save=True)

# 自己构造一些数据
# 在识别自己的数据时，传入classes = [0,2]
# 代表只输出0和2，也就是安全帽是否佩戴这俩类别
# line_width = 30 表示指定识别框的字体的大小为30
# model.predict('.\\test2.png', save=True, classes=[0,2], line_width=30)
# model.predict('.\\test3.png', save=True, classes=[0,2], line_width=30)
# model.predict('myself1.mp4', save=True, classes=[0,2], line_width=30)
# model.predict('myself2.mp4', save=True, classes=[0,2], line_width=30)



