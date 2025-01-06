Ship Instance Segmentation Network for SAR and Optical Images Based on MobileViT-v3

Ship instance segmentation is pivotal for various maritime applications, yet it remains challenging due to the complex ocean environment and diverse ship characteristics.  This paper introduces MDFNet, a novel single-stage instance segmentation network based on MobileViT-v3, which effectively integrates convolutional and Transformer mechanisms.  The backbone of MDFNet incorporates the CpnMvitbv3 module, leveraging MobileViT-v3 to enhance the understanding of spatial relationships between objects through global attention.  The neck network employs DODConv with dynamic weights to handle complex backgrounds and multi-scale objects.  Furthermore, we introduce Focal-SIoU loss to improve small object detection, manage challenging samples, and optimize boundary precision.  Experimental results on the SSDD (SAR) and MarShipInsSeg (visible light) datasets demonstrate that MDFNet achieves state-of-the-art performance with an average precision of 63.8% on SSDD and 49.4% on MarShipInsSeg, surpassing existing methods by significant margins.  Notably, MDFNet maintains a compact model size of 24MB, showcasing its efficiency and adaptability across different datasets and environments.

Environment dependencies and core packages: windows11 system, pytorch=1.12.0, python=3.8,numpy=1.23.2,mmcv=1.6.2,timm=0.6.7, opencv-python=4.10.0.84

Run python train_seg.py --cfg ultralytics\cfg\models\cfg2024\YOLOv8-Seg\yyyy.yaml directly on the terminal to start training

If you have any questions, please contact us at：m230200742@st.shou.edu.cn;

