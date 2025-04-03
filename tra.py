from ultralytics import YOLO

# 1. Load a pre-trained model for transfer learning instead of starting from scratch
model = YOLO("yolov8n.pt")  

# 2. Enhanced training with more parameters and optimizations
results = model.train(
    data="input/data.yaml",      # Your dataset configuration
    epochs=100,                  # Increase epochs for better convergence
    imgsz=640,                   # Input image size
    batch=16,                    # Adjust batch size based on your GPU memory
    patience=20,                 # Early stopping patience
    lr0=0.01,                    # Initial learning rate
    lrf=0.001,                   # Final learning rate
    optimizer="Adam",            # Try different optimizers
    weight_decay=0.0005,         # Weight decay for regularization
    cos_lr=True,                 # Use cosine learning rate scheduler
    augment=True,                # Use data augmentation
    mixup=0.1,                   # Apply mixup augmentation
    degrees=10.0,                # Maximum rotation for augmentation
    translate=0.1,               # Maximum translation for augmentation
    scale=0.5,                   # Maximum scale for augmentation
    fliplr=0.5,                  # Horizontal flip probability
    mosaic=1.0,                  # Mosaic augmentation probability
    workers=8,                   # Number of worker threads
    save=True,                   # Save checkpoints
    save_period=5,               # Save checkpoints every 5 epochs
    device="cpu",                    # GPU device (use comma-separated for multiple GPUs)
    cache=True,                  # Cache images in RAM
)

# 3. Evaluate the model on validation set
model.val()

# 4. Export the model for deployment (optional)
model.export(format="onnx")  # Or other formats like TensorRT, CoreML, etc.