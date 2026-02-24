This project implements a real-time object detection system using the YOLOv5 deep learning model deployed on a PYNQ-Z2 FPGA board. The system demonstrates both CPU-based inference and hardware-accelerated inference using FPGA/DPU architecture for improved performance and efficiency.

The YOLOv5 model was converted to ONNX format and executed using OpenCV DNN for baseline CPU benchmarking. The same model was then deployed onto the FPGA accelerator to evaluate speedup and performance gains achieved through hardware acceleration.
