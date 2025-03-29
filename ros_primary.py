import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
import onnxruntime as ort
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

class YOLOv9ROS2(Node):
    def __init__(self):
        super().__init__('yolov9_ros2')

        # Declare and get parameters
        self.declare_parameter('onnx_model_path', 'runs/train/exp1/weights/best.onnx')
        self.onnx_path = self.get_parameter('onnx_model_path').get_parameter_value().string_value

        # Load ONNX model
        self.load_model()

        # ROS2 Subscribers (ZED2i RGB and Depth Images)
        self.bridge = CvBridge()
        self.image_sub = Subscriber(self, Image, '/zed2i/zed_node/rgb/image_rect_color')
        self.depth_sub = Subscriber(self, Image, '/zed2i/zed_node/depth/depth_registered')

        # Synchronize messages
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.image_callback)

        self.get_logger().info("YOLOv9 ROS2 node initialized.")

    def load_model(self):
        """Load ONNX model with GPU acceleration (if available)."""
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
            self.get_logger().info(f"Loaded ONNX model: {self.onnx_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model: {e}")
            raise RuntimeError("ONNX model loading failed.")

    def image_callback(self, img_msg, depth_msg):
        """Callback to process synchronized RGB and depth images."""
        try:
            # Convert ROS2 images to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")  # Float32 depth map

            # Preprocess image for YOLOv9
            input_tensor = self.preprocess_image(cv_image)

            # Run inference
            detections = self.run_inference(input_tensor)

            # Draw bounding boxes and estimate depth
            if detections is not None:
                cv_image = self.draw_bounding_boxes(cv_image, depth_image, detections)

            # Display image
            cv2.imshow("YOLOv9 Detections", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

    def preprocess_image(self, image):
        """Preprocess image for YOLOv9 ONNX model."""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))  # Resize to YOLOv9 input size
        img = img.astype(np.float32) / 255.0  # Normalize
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)  # (1,3,640,640)
        return img

    def run_inference(self, input_tensor):
        """Run YOLOv9 inference using ONNX model."""
        try:
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
            ort_inputs = {self.session.get_inputs()[0].name: input_tensor.numpy()}
            output = self.session.run(None, ort_inputs)[0]
            return self.non_max_suppression(output)
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
            return None

    def non_max_suppression(self, predictions, conf_threshold=0.25, iou_threshold=0.45):
        """Apply Non-Maximum Suppression (NMS) to filter detections."""
        predictions = torch.tensor(predictions) if not isinstance(predictions, torch.Tensor) else predictions
        valid_detections = []

        for pred in predictions:
            conf_mask = pred[:, 4] > conf_threshold  # Confidence threshold
            pred = pred[conf_mask]

            if not len(pred):
                continue

            # Apply NMS
            keep = torch.ops.torchvision.nms(pred[:, :4], pred[:, 4], iou_threshold)
            valid_detections.append(pred[keep].tolist())

        return valid_detections if valid_detections else None

    def draw_bounding_boxes(self, image, depth_image, detections):
        """Draw bounding boxes and estimate object distances."""
        if not detections:
            return image  # No detections

        for det in detections:
            for d in det:
                if len(d) != 6:
                    self.get_logger().warn(f"Unexpected detection format: {d}")
                    continue

                x1, y1, x2, y2, conf, cls = map(int, d[:6])  # Ensure integers
                label = f'Class {cls} | Conf {conf:.2f}'

                # Estimate distance using depth image
                depth_value = depth_image[y1 + (y2 - y1) // 2, x1 + (x2 - x1) // 2]
                depth_text = f'Depth: {depth_value:.2f}m' if depth_value > 0 else "No depth"

                # Draw bounding box and labels
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, depth_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return image

def main(args=None):
    """Initialize and run the ROS 2 node."""
    rclpy.init(args=args)
    node = YOLOv9ROS2()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()  # Cleanup OpenCV windows
        rclpy.shutdown()

if __name__ == '__main__':
    main()



