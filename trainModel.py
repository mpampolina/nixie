from ultralytics import YOLO

# Load the YOLOv8 nano model from ultralytics
model = YOLO("yolov8n.pt")

results = model.train(data="config.yaml", epochs=1)
model.export()