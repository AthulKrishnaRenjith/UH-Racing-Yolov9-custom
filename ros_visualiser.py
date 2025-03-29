import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import torch
from utils.general import non_max_suppression, scale_boxes
from message_filters import ApproximateTimeSynchronizer, Subscriber

class YOLOv9ONNXDepthViewer(Node):
    def __init__(self):
        super().__init__('yolo_v9_onnx_depth_viewer')
        self.bridge = CvBridge()

        self.left_sub_ = Subscriber(self, Image, '/zed2i/zed_node/left/image_rect_color')
        self.depth_sub_ = Subscriber(self, Image, '/zed2i/zed_node/depth/depth_registered')
        self.camera_info_sub = self.create_subscription(CameraInfo, '/zed2i/zed_node/depth/camera_info', self.camera_info_callback, 10)

        self.ats_ = ApproximateTimeSynchronizer([self.left_sub_, self.depth_sub_], 10, 0.1)
        self.ats_.registerCallback(self.image_callback)

        self.publisher = self.create_publisher(String, '/detections', 10)

        self.onnx_path = 'runs/train/exp1/weights/best.onnx'
        try:
            providers = ['CUDAExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        except Exception as e:
            self.get_logger().error(f'Failed to load ONNX model: {e}')
            raise
        
        self.imgsz = (640, 640)
        self.get_logger().info('YOLOv9 ONNX Depth Viewer node initialized')

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.imgsz, interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float16) / 255.0
        return img

    def draw_bounding_boxes(self, img, pred):
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(self.imgsz, det[:, :4], img.shape).round()
                for *xyxy, conf, cls in det:
                    # Draw bounding box
                    xyxy = list(map(int, xyxy))
                    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                    # Display class and confidence label
                    label = f'Class {int(cls)}: {conf:.2f}'
                    cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

    def image_callback(self, left_img_msg, depth_img_msg):
        self.get_logger().info('Received synchronized images')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        img = self.preprocess(cv_image)

        try:
            inputs = {self.session.get_inputs()[0].name: img}
            output = self.session.run(None, inputs)[0]
            pred = torch.tensor(output, dtype=torch.float16)
            pred = non_max_suppression(pred, 0.25, 0.45)
        except Exception as e:
            self.get_logger().error(f'Error during inference: {e}')
            return

        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(self.imgsz, det[:, :4], cv_image.shape).round()
                for *xyxy, conf, cls in det:
                    detections.append(f'Class: {int(cls)}, Confidence: {conf:.2f}, BBox: {list(map(int, xyxy))}')

        # Draw bounding boxes on the image
        output_img = self.draw_bounding_boxes(cv_image.copy(), pred)

        # Visualize both input and output
        cv2.imshow("Input Image", cv_image)  # Show original input image
        cv2.imshow("Output Image", output_img)  # Show output image with detections

        cv2.waitKey(1)  # Refresh window

        detection_msg = String()
        detection_msg.data = '\n'.join(detections)
        self.publisher.publish(detection_msg)
        self.get_logger().info(f'Published {len(detections)} detections')

    def camera_info_callback(self, msg):
        self.get_logger().info('Received camera info')
        self.camera_info = msg


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv9ONNXDepthViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

