import sys
import argparse
import os

sys.path.append(r'E:\MDFNet') # Path

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml)#.load('E:\daima/ultralyticsPro1020\yolov8n-seg.pt')

    model.info()

    # 实例分割训练
    results = model.train(data='E:/MDFNet/ultralytics\cfg\datasets\coco128-seg.yaml',epochs=150,imgsz=640,  workers=8, batch=8)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'E:\MDFNet\ultralytics\cfg\models\cfg2024\YOLOv8-Seg\yyyy.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
