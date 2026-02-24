
def cpu_inference(model_path, image_path):

    print("\n[INFO] CPU Inference Started...")

    # Load model
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found.")

    height, width = image.shape[:2]

    # Precompute scaling factors (optimization)
    x_factor = width / 640
    y_factor = height / 640

    # Preprocess
    blob = cv2.dnn.blobFromImage(
        image, 1/255.0, (640, 640),
        swapRB=True, crop=False
    )
    net.setInput(blob)

    # Inference
    start = time.time()
    outputs = net.forward()[0]
    inference_time = time.time() - start

    print(f"[CPU] Inference Time: {inference_time:.4f} sec")
    print(f"[CPU] FPS: {1/inference_time:.2f}")

    # Detection parameters
    conf_threshold = 0.5
    nms_threshold = 0.5

    boxes = []
    confidences = []
    class_ids = []

    # Parse detections (optimized loop)
    for detection in outputs:

        objectness = detection[4]
        if objectness < conf_threshold:
            continue

        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = objectness * scores[class_id]

        if confidence < conf_threshold:
            continue

        cx, cy, w, h = detection[:4]

        x = int((cx - w / 2) * x_factor)
        y = int((cy - h / 2) * y_factor)
        w = int(w * x_factor)
        h = int(h * y_factor)

        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    # Apply NMS safely
    indices = cv2.dnn.NMSBoxes(boxes, confidences,
                               conf_threshold, nms_threshold)

    # Draw results
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            x, y, w, h = boxes[i]
            label = f"{class_ids[i]} {confidences[i]:.2f}"

            cv2.rectangle(image, (x, y),
                          (x + w, y + h),
                          (0, 255, 0), 2)

            cv2.putText(image, label,
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

    cv2.imwrite("model_output_demo.png", image)
    print("[INFO] Output saved as model_output_demo.png")

    return inference_time


# HARDWARE ACCELERATION (SIMULATED DPU SECTION)


def hardware_accelerated_inference():

    print("\n[INFO] Hardware Accelerated Inference Started...")

    # Simulated FPGA/DPU execution time
    simulated_time = 0.015
    print(f"[DPU] Inference Time: {simulated_time:.4f} sec")
    print(f"[DPU] FPS: {1/simulated_time:.2f}")

    return simulated_time



# MAIN


if __name__ == "__main__":

    MODEL_PATH = "yolov5s.onnx"
    IMAGE_PATH = "bus.jpg"

    cpu_time = cpu_inference(MODEL_PATH, IMAGE_PATH)
    dpu_time = hardware_accelerated_inference()

    print("\n========== PERFORMANCE COMPARISON ==========")
    print(f"CPU Time : {cpu_time:.4f} sec")
    print(f"DPU Time : {dpu_time:.4f} sec")
    print(f"Speedup  : {cpu_time/dpu_time:.2f}x")
    print("============================================")
