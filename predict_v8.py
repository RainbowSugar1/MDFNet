import sys
import argparse
import os


from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml) 

    model.info()

    model = YOLO(r"G:\实验结果\消融实验ssdd\12backbone+loss+conv\train13\weights\best.pt") # 权重路径

    # 检测图像路径"G:\实验结果\实验87\train12\weights\best.pt"
    results=model.predict(r"G:\可视化image\ssdd\000641.jpg",save=True, imgsz=640, conf=0.5, save_txt=True, save_conf=True)
    results[0].show()



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r"G:\实验结果\消融实验ssdd\12backbone+loss+conv\train13\weights\best.pt", help='initial weights path')
    parser.add_argument('--artifact_alias', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)