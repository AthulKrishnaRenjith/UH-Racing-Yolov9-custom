#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import torch
from utils.general import non_max_suppression, scale_boxes
from message_filters import Subscriber, ApproximateTimeSynchronizer
from eufs_msgs.msg import ConeWithCovariance, ConeArrayWithCovariance
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class RacingConeDetectionNode(Node):
    def __init__(self):
        super().__init__('racing_cone_detection_node')
        self.bridge = CvBridge()
        
        # Cone definitions
        self.cone_classes = {
            0: 'large_orange_cone',
            1: 'orange_cone', 
            2: 'blue_cone',
            3: 'yellow_cone'
        }
        
        self.cone_colors = {
            0: (0, 140, 255),  # BGR - Dark Orange
            1: (0, 165, 255),  # BGR - Orange
            2: (255, 0, 0),     # BGR - Blue
            3: (0, 255, 255)    # BGR - Yellow
        }

        # Model configuration
        self.imgsz = (640, 640)
        self.onnx_path = 'runs/train/exp1/weights/best.onnx'
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        
        # Initialize ONNX Runtime with float16 support
        try:
            providers = ['CUDAExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(self.onnx_path, 
                                             providers=providers,
                                             sess_options=so)
            self.get_logger().info(f"Loaded ONNX model from {self.onnx_path}")
            # Check model input type
            model_input_type = self.session.get_inputs()[0].type
            self.get_logger().info(f"Model expects input type: {model_input_type}")
            self.use_fp16 = 'float16' in model_input_type
        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model: {e}")
            raise
        
        # Setup synchronized subscribers
        self.left_image_sub = Subscriber(self, Image, '/zed2i/zed_node/left/image_rect_color')
        self.depth_sub = Subscriber(self, Image, '/zed2i/zed_node/depth/depth_registered')
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/zed2i/zed_node/depth/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Time synchronizer
        self.ats = ApproximateTimeSynchronizer(
            [self.left_image_sub, self.depth_sub],
            10,
            0.1
        )
        self.ats.registerCallback(self.sync_callback)
        
        # Setup publishers
        self.detection_pub = self.create_publisher(Float32MultiArray, '/cone_detections', 10)
        self.visualization_pub = self.create_publisher(Image, '/cone_detections_visualization', 10)
        self.cones_3d_pub = self.create_publisher(ConeArrayWithCovariance, '/cones_3d', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/cone_markers', 10)
        
        # Camera info storage
        self.camera_info = None
        self.cv_depth = None
        
        self.get_logger().info("Node initialized and ready")

    def preprocess(self, img):
        """Preprocess image for ONNX model inference with proper dtype"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.imgsz, interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)
        
        # Convert to float16 if model expects it, otherwise float32
        if self.use_fp16:
            img = img.astype(np.float16) / 255.0
        else:
            img = img.astype(np.float32) / 255.0
            
        return img

    def camera_info_callback(self, msg):
        """Store camera intrinsic parameters"""
        self.camera_info = msg
        self.get_logger().info("Received camera info", throttle_duration_sec=5)

    def sync_callback(self, image_msg, depth_msg):
        """Process synchronized image and depth data"""
        try:
            # Convert ROS messages to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            self.cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
            
            # Run detection with proper dtype handling
            img_tensor = self.preprocess(cv_image)
            inputs = {self.session.get_inputs()[0].name: img_tensor}
            
            # Run inference
            outputs = self.session.run(None, inputs)
            
            # Convert outputs to float32 for consistency in post-processing
            if outputs[0].dtype == np.float16:
                pred = torch.tensor(outputs[0], dtype=torch.float32)
            else:
                pred = torch.tensor(outputs[0])
                
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            
            # Process and visualize results
            self.process_detections(pred, cv_image.copy())
            
        except Exception as e:
            self.get_logger().error(f"Processing failed: {str(e)}")

    def process_detections(self, pred, img):
        """Process detections and calculate 3D positions"""
        detections = Float32MultiArray()
        dim = MultiArrayDimension()
        dim.label = "detections"
        dim.stride = 6  # x1,y1,x2,y2,class,conf
        detections.layout.dim.append(dim)
        
        annotated_img = img.copy()
        detection_data = []
        cones_3d = []
        
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(self.imgsz, det[:, :4], img.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    cls = int(cls)
                    if cls not in self.cone_classes:
                        continue
                        
                    # Add to detection data
                    detection_data.extend([
                        float(xyxy[0]), float(xyxy[1]),
                        float(xyxy[2]), float(xyxy[3]),
                        float(cls), float(conf)
                    ])
                    
                    # Calculate 3D position if we have depth
                    if self.cv_depth is not None and self.camera_info is not None:
                        xmin, ymin, xmax, ymax = map(int, xyxy)
                        depth_values = self.cv_depth[ymin:ymax, xmin:xmax]
                        valid_depths = depth_values[~np.isnan(depth_values)]
                        
                        if len(valid_depths) > 0:
                            depth = np.median(valid_depths)
                            
                            # Calculate 3D coordinates (simplified projection)
                            u = (xmin + xmax) / 2
                            v = (ymin + ymax) / 2
                            
                            X = depth
                            Y = (u - self.camera_info.p[2]) / self.camera_info.p[0] * X
                            Z = (v - self.camera_info.p[6]) / self.camera_info.p[5] * X
                            
                            if np.isfinite([X, Y, Z]).all():
                                cones_3d.append((X, Y, self.cone_classes[cls]))
                    
                    # Draw detection
                    self.draw_detection(annotated_img, xyxy, cls, conf)
        
        # Publish all messages
        detections.layout.dim[0].size = len(detection_data) // 6
        detections.data = detection_data
        self.detection_pub.publish(detections)
        self.visualization_pub.publish(self.bridge.cv2_to_imgmsg(annotated_img, 'bgr8'))
        
        if cones_3d:
            self.publish_3d_cones(cones_3d)
        
        if detection_data:
            self.get_logger().info(f"Detected {len(detection_data)//6} cones", throttle_duration_sec=1)

    def draw_detection(self, img, xyxy, cls, conf):
        """Draw a single detection on the image"""
        label = f"{self.cone_classes[cls]} {conf:.2f}"
        color = self.cone_colors[cls]
        
        # Draw bounding box
        cv2.rectangle(
            img,
            (int(xyxy[0]), int(xyxy[1])),
            (int(xyxy[2]), int(xyxy[3])),
            color, 2
        )
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(
            img,
            (int(xyxy[0]), int(xyxy[1]) - h - 4),
            (int(xyxy[0]) + w, int(xyxy[1])),
            color, -1
        )
        
        # Draw label text
        cv2.putText(
            img, label,
            (int(xyxy[0]), int(xyxy[1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    def publish_3d_cones(self, cones):
        """Publish 3D cone positions and visualization markers"""
        cone_array = ConeArrayWithCovariance()
        cone_array.header.stamp = self.get_clock().now().to_msg()
        cone_array.header.frame_id = "base_link"
    
        marker_array = MarkerArray()
    
    # Clear previous markers
        delete_marker = Marker()
        delete_marker.header.frame_id = "base_link"
        delete_marker.header.stamp = cone_array.header.stamp
        delete_marker.ns = "cones"
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
    
        for i, (x, y, color) in enumerate(cones):
        # Create cone message
            cone_msg = ConeWithCovariance()
            cone_msg.point.x = float(x)
            cone_msg.point.y = float(y)
            cone_msg.point.z = 0.0
        
        # Color mapping (converted to 0-1 range for RViz)
            if color == 'blue_cone':
                cone_array.blue_cones.append(cone_msg)
                marker_color = (0.0, 0.0, 1.0)  # Blue in RGB (0-1)
            elif color == 'yellow_cone':
                cone_array.yellow_cones.append(cone_msg)
                marker_color = (1.0, 1.0, 0.0)  # Yellow in RGB (0-1)
            elif color == 'orange_cone':
                cone_array.orange_cones.append(cone_msg)
                marker_color = (1.0, 0.65, 0.0)  # Orange in RGB (0-1)
            elif color == 'large_orange_cone':
                cone_array.big_orange_cones.append(cone_msg)
                marker_color = (1.0, 0.55, 0.0)  # Dark Orange in RGB (0-1)
        
        # Create visualization marker
            marker = Marker()
            marker.header = cone_array.header
            marker.ns = "cones"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.75
            marker.color.r = float(marker_color[0])  # Explicit float conversion
            marker.color.g = float(marker_color[1])
            marker.color.b = float(marker_color[2])
            marker.color.a = 1.0  # Fully opaque
            marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
            marker_array.markers.append(marker)
    
        self.cones_3d_pub.publish(cone_array)
        self.marker_pub.publish(marker_array)

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
