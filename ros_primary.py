#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from pathlib import Path

# YOLO imports
from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.augmentations import letterbox

class RacingConeDetectionNode(Node):
    def __init__(self):
        super().__init__('racing_cone_detection_node')
        self.bridge = CvBridge()
        
        # Initialize model attributes
        self.pt = None
        self.stride = None
        self.model = None
        
        # Cone definitions
        self.cone_classes = {
            0: 'large_orange_cone',
            1: 'orange_cone', 
            2: 'blue_cone',
            3: 'yellow_cone'
        }
        
        self.cone_colors = {
            0: (0, 140, 255),
            1: (0, 165, 255),
            2: (255, 0, 0),
            3: (0, 255, 255)
        }

        # Fixed parameters matching training
        self.model_path = 'runs/train/exp1/weights/best.pt'
        self.imgsz = (640, 640)
        
        # Declare parameters with correct types
        self.declare_parameter('weights', self.model_path)
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('max_det', 100)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('half', False)
        
        # Get parameters correctly
        self.weights = self.get_parameter('weights').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.iou_thres = self.get_parameter('iou_thres').value
        self.max_det = self.get_parameter('max_det').value
        self.device = self.get_parameter('device').value
        self.half = self.get_parameter('half').value
        
        # Setup subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/zed2i/zed_node/left/image_rect_color',
            self.image_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/zed2i/zed_node/depth/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Setup publishers
        self.detection_pub = self.create_publisher(Float32MultiArray, '/cone_detections', 10)
        self.visualization_pub = self.create_publisher(Image, '/cone_detections_visualization', 10)
        
        # Load model
        self.load_model()
        self.get_logger().info("Node initialized with 640x640 processing")

    def load_model(self):
        """Proper model loading with error handling"""
        try:
            self.device = select_device(self.device)
            self.model = DetectMultiBackend(
                self.weights,
                device=self.device,
                dnn=False,
                data=None,
                fp16=self.half
            )
            
            # Set critical attributes
            self.stride = self.model.stride
            self.pt = self.model.pt
            self.imgsz = check_img_size(self.imgsz, s=self.stride)
            
            # Warmup
            self.model.warmup(imgsz=(1 if self.pt else 1, 3, *self.imgsz))
            
            self.get_logger().info(
                f"Model loaded | stride={self.stride} | pt={self.pt} | "
                f"imgsz={self.imgsz}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Model loading failed: {str(e)}")
            raise

    def preprocess(self, img):
        """Consistent 640x640 preprocessing"""
        img = letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # Matches training
        img /= 255.0
        return img.unsqueeze(0)  # Add batch dim

    def image_callback(self, msg):
        """Robust image processing pipeline"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            img_tensor = self.preprocess(cv_image)
            
            with torch.no_grad():
                pred = self.model(img_tensor)[0]
                pred = non_max_suppression(
                    pred,
                    self.conf_thres,
                    self.iou_thres,
                    classes=None,
                    agnostic_nms=False,
                    max_det=self.max_det
                )
            
            self.process_detections(pred, cv_image.copy())
            
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {str(e)}", throttle_duration_sec=1)

    def process_detections(self, pred, img):
        """Convert detections to ROS messages"""
        detections = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label = "detections"
        dim.stride = 6  # x1,y1,x2,y2,class,conf
        detections.layout.dim.append(dim)
        
        annotated = img.copy()
        data = []
        
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(self.imgsz, det[:, :4], img.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    cls = int(cls)
                    if cls not in self.cone_classes:
                        continue
                        
                    data.extend([
                        float(xyxy[0]), float(xyxy[1]),
                        float(xyxy[2]), float(xyxy[3]),
                        float(cls), float(conf)
                    ])
                    
                    # Draw detection
                    label = f"{self.cone_classes[cls]} {conf:.2f}"
                    color = self.cone_colors[cls]
                    
                    cv2.rectangle(
                        annotated,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        color, 2
                    )
                    
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(
                        annotated,
                        (int(xyxy[0]), int(xyxy[1]) - h - 4),
                        (int(xyxy[0]) + w, int(xyxy[1])),
                        color, -1
                    )
                    
                    cv2.putText(
                        annotated, label,
                        (int(xyxy[0]), int(xyxy[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA
                    )
        
        detections.layout.dim[0].size = len(data) // 6
        detections.data = data
        
        self.detection_pub.publish(detections)
        self.visualization_pub.publish(
            self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        )
        
        if data:
            self.get_logger().info(
                f"Detected {len(data)//6} cones",
                throttle_duration_sec=1
            )

def main(args=None):
    rclpy.init(args=args)
    node = RacingConeDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
