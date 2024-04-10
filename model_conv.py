from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n-pose.pt')

# Export the model to TensorRT format
model.export(format='engine')  # format='engine'으로 지정하면 TensorRT 사용

# Load the exported TensorRT model
tensorrt_model = YOLO('yolov8n-pose.engine')

# Run inference
results = tensorrt_model('https://ultralytics.com/images/bus.jpg')
