from ultralytics import YOLO

model = YOLO("models/dermavision_best_v3.pt")

print(model.task)
print(model.names)