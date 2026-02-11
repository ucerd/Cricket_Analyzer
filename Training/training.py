from ultralytics import YOLO
import torch

# ==================================================
# CONFIGURATION
# ==================================================
DATA_YAML = "dataset.yaml"
MODEL = "yolov11m.pt"

IMG_SIZE = 640
EPOCHS = 150
BATCH = 16
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ==================================================
# LOAD MODEL
# ==================================================
model = YOLO(MODEL)

# ==================================================
# TRAIN
# ==================================================
model.train(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH,
    device=DEVICE,
    
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    
    hsv_h=0.02,
    hsv_s=0.7,
    hsv_v=0.4,
    mosaic=0.2,
    close_mosaic=10,
    degrees=3,
    translate=0.05,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    
    patience=20,
    save=True,
    cache=True,
    workers=8,
    
    project="runs/cricket",
    name="cricket_yolov11",
    exist_ok=True,
    verbose=True
)

print("Training finished. Weights saved.")
