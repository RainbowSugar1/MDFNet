import sys
import argparse
import os


from ultralytics import YOLO

def main(opt):
    #yaml = opt.cfg
   # model = YOLO(yaml)

    #model.info()

    model = YOLO(r"E:\daima\ultralyticsPro1020\runs\segment\train2\weights\best.pt") # 权重路径

    # 检测图像路径"G:\实验结果\实验87\train12\weights\best.pt"
    model.val(data='E:\daima/ultralyticsPro1020/ultralytics\cfg\datasets\coco128-seg.yaml',
              split='val',
              imgsz=640,
              workers=4,
              batch=4,
              project='runs/val',
              name='exp',
              )




def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='initial weights path')
    parser.add_argument('--artifact_alias', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


