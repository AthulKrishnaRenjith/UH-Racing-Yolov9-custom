import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This is needed for initializing CUDA driver

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load engine
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        print("Loading engine...")
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    input_binding = engine.get_tensor_name(0)  # Use get_tensor_name to avoid deprecation warnings
    output_binding = engine.get_tensor_name(1)  # Use get_tensor_name to avoid deprecation warnings

    input_shape = engine.get_tensor_shape(input_binding)
    output_shape = engine.get_tensor_shape(output_binding)

    print(f"Input shape: {input_shape}, Output shape: {output_shape}")

    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float16)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float16)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    return h_input, d_input, h_output, d_output

def preprocess_image(img, input_shape=(640, 640)):
    img_resized = cv2.resize(img, input_shape)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(img_transposed, axis=0).ravel()

def post_process(output, input_shape, conf_threshold=0.5, iou_threshold=0.4):
    print(f"Output shape before post-processing: {output.shape}")
    num_detections = output.shape[0]
    boxes = []
    class_ids = []
    confidences = []

    # Adjust this part if necessary depending on output format
    for i in range(num_detections):
        box = output[i, :4]
        class_id = int(output[i, 4])
        confidence = output[i, 5]

        if confidence >= conf_threshold:
            box[0] *= input_shape[1]
            box[1] *= input_shape[0]
            box[2] *= input_shape[1]
            box[3] *= input_shape[0]

            boxes.append(box)
            class_ids.append(class_id)
            confidences.append(confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

    final_boxes = []
    final_class_ids = []
    final_confidences = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_class_ids.append(class_ids[i])
            final_confidences.append(confidences[i])

    return final_boxes, final_class_ids, final_confidences

def infer(engine_path="best_fp16.engine", image_path="test.jpg", input_shape=(640, 640)):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    h_input, d_input, h_output, d_output = allocate_buffers(engine)

    # Load and preprocess image
    image = cv2.imread(image_path)
    input_data = preprocess_image(image, input_shape)
    np.copyto(h_input, input_data)

    # Run inference
    print("Running inference...")
    cuda.memcpy_htod(d_input, h_input)
    context.execute_v2([int(d_input), int(d_output)])
    cuda.memcpy_dtoh(h_output, d_output)

    # Inspect output tensor for debugging
    print(f"Output tensor (first 10 elements): {h_output[:10]}")

    # Post-process output
    detections = post_process(h_output, input_shape)
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = f"{int(cls)} {conf:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display
    cv2.imshow("Inference", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    infer("best_fp16.engine", "test.jpg")

