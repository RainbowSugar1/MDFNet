from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('E:\daima\yolov8-seg1/runs\segment/train5\weights/best.pt') # 自己训练结束后的模型权重
    model.val(data='E:\daima\yolov8-seg1/ultralytics\cfg\datasets\coco128-seg.yaml',
              split='val',
              imgsz=640,
              batch=2,
              save_json=True, # if you need to cal coco metrice
              project='runs/val1',
              name='exp',
              )
