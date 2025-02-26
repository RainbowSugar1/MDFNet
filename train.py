from ultralytics import YOLO

# Load a model
model = YOLO("E:\daima\yolov8-seg1/ultralytics\cfg\models/v8\yolov8s-seg.yaml")  # build a new model from YAML
#model = YOLO("E:\daima\yolov8-seg1\yolov8n.pt")  # load a pretrained model (recommended for training)
#model = YOLO("E:\daima\yolov8seg/ultralytics-main/ultralytics/cfg/models/v8\yolov8n-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
if __name__ == '__main__':
   results = model.train(data="E:\daima\yolov8-seg1/ultralytics\cfg\datasets\coco128-seg.yaml", epochs=150, imgsz=640)