from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n-seg.pt")  # load an official model
model = YOLO("G:\实验结果\实验83/train19\weights/best.pt")  # load a custom model

# Predict with the model
results = model("E:\daima/ultralyticsPro1020\images/000089.jpg")  # predict on an image
results[0].show()