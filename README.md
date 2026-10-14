Coconut Disease Detection Using YOLOv8
1. Project Overview
Coconut Disease Detection is a computer vision project designed to identify coconut diseases and pests from images, videos, and live webcam footage using a trained YOLOv8 object detection model.
Coconut crops can be affected by various diseases and pests that impact plant health, growth, and productivity. Manual inspection of coconut trees can be time-consuming, particularly when monitoring large plantations. This project explores how deep learning and computer vision can assist in identifying visual symptoms and highlighting potentially affected areas.
The application provides an interactive interface built with Streamlit, allowing users to upload images or videos and view detection results with bounding boxes, class labels, and confidence scores.
The system is intended as a decision-support tool. Its predictions should be verified by agricultural experts before taking any treatment or crop-management decisions.
2. Project Objectives
- Detect trained coconut disease and pest categories using computer vision.
- Automate visual inspection using a YOLOv8 object detection model.
- Provide image, video, and real-time webcam detection.
- Display detected objects with bounding boxes, class labels, and confidence scores.
- Apply image enhancement techniques to improve visual contrast.
- Track detected objects across video frames.
- Summarize recurring detections in uploaded videos.
- Provide a training pipeline for fine-tuning the model on a labelled dataset.
3. Key Features
3.1 Image Recognition
The application allows users to upload images in JPG, JPEG, or PNG format.
Features include:
- Upload images through the Streamlit interface.
- Display the original image.
- Apply image enhancement for visual inspection.
- Detect objects using the trained YOLOv8 model.
- Display bounding boxes around detected objects.
- Show predicted class names and confidence scores.
3.2 Video Recognition
The application supports video-based detection using MP4, AVI, and MOV files.
Features include:
- Upload and preview video files.
- Process approximately one frame per second to reduce computational requirements.
- Display processing progress and selected annotated frames.
- Track detections across processed frames.
- Summarize detections by predicted class.
- Calculate average confidence scores for detected categories.
This feature can help users review visual patterns across multiple video frames instead of manually inspecting every frame.
3.3 Live Webcam Detection
The application supports real-time detection through a computer's webcam.
Features include:
- Capture live video frames.
- Perform object detection on captured frames.
- Display bounding boxes and predicted class labels.
- Show confidence scores and tracking IDs.
- Display an approximate frames-per-second (FPS) indicator.
- Maintain basic detection statistics during the session.
Webcam functionality depends on device permissions, camera availability, and the environment in which Streamlit is running.
3.4 Image Enhancement
The application uses image enhancement techniques to improve contrast before visual analysis.
The enhancement pipeline includes:
1. Converting the image from RGB to LAB colour space.
2. Applying Contrast Limited Adaptive Histogram Equalization (CLAHE) to the lightness channel.
3. Converting the enhanced image back to RGB.
4. Applying Gaussian blur to reduce some high-frequency noise.
Image enhancement may improve visibility in certain conditions, but it does not guarantee higher detection accuracy.
3.5 Object Tracking
The application implements a basic tracking mechanism for video and webcam detection.
The tracker uses Intersection over Union (IoU) to associate detected bounding boxes across frames. Matched detections receive tracking IDs, while bounding-box positions and confidence scores are smoothed over time.
Tracks that remain unmatched for a specified number of updates are removed.
This is a lightweight tracking approach rather than a dedicated multi-object tracking model, so tracking IDs may become inconsistent when objects overlap or move rapidly.
4. Technology Stack
- Python: Core programming language.
- YOLOv8: Deep-learning model for object detection.
- Ultralytics: Framework for loading, training, validating, and exporting YOLO models.
- OpenCV: Image processing, video handling, webcam access, and bounding-box visualization.
- Streamlit: Interactive web application interface.
- NumPy: Numerical computation and image-array processing.
- Pillow (PIL): Image loading and manipulation.
- PyTorch: Deep-learning runtime used by Ultralytics.
5. Project Structure
The supplied project contains the following files:
COCONUT-DISEASE-DETECTION-main/
│
├── best.pt
├── newapp.py
└── tra.py


File Descriptions
best.pt
Contains the trained YOLO model weights used for object detection. The categories supported by the model depend on the dataset used during training.
newapp.py
Contains the Streamlit application, including image upload, video processing, webcam detection, image enhancement, object tracking, and visualization.
tra.py
Contains the model-training pipeline for training or fine-tuning YOLOv8, validating the model, and exporting it to ONNX format.
Additional Files Required
The training script references input/data.yaml, which is not included in the supplied archive. The dataset images and labels are also required to train the model.
The application expects its model at runs/detect/train3/weights/best.pt, whereas the supplied best.pt is located in the project root. This path must be corrected before running the application unless the expected file already exists.
6. How the System Works
The application follows this general workflow:
1. The user launches the Streamlit application.
2. The trained YOLOv8 model is loaded.
3. The user selects image recognition, video recognition, or live detection.
4. The input is processed and optionally enhanced.
5. YOLOv8 predicts object locations, class IDs, and confidence scores.
6. Predictions are filtered according to the confidence threshold.
7. Bounding boxes and class labels are displayed.
8. For video and webcam inputs, detections may be tracked across frames.
9. For uploaded videos, the application generates a summary of recurring detections.
YOLOv8 Object Detection
YOLOv8 is an object detection architecture that predicts the locations and classes of objects within an image.
The model produces bounding boxes, class predictions, and confidence scores. The trained best.pt file contains the learned parameters from a previous training run.
The exact disease and pest categories depend on the training dataset and model metadata. They should be inspected rather than assumed.
7. Installation and Setup
Step 1: Install Python
Install a compatible Python version. Python 3.10 or 3.11 is a practical starting point for this project, although package compatibility depends on the operating system and installed libraries.
Step 2: Extract the Project
Extract COCONUT-DISEASE-DETECTION-main.zip and open a terminal in the extracted project directory.
Step 3: Create a Virtual Environment
For Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1


For Windows Command Prompt:
python -m venv .venv
.venv\Scripts\activate.bat


For Linux or macOS:
python3 -m venv .venv
source .venv/bin/activate


Step 4: Install Dependencies
Run:
python -m pip install --upgrade pip
pip install streamlit ultralytics opencv-python numpy Pillow


PyTorch is generally installed as a dependency of Ultralytics. If you need GPU acceleration, install a compatible PyTorch build for your CUDA and hardware configuration.
Step 5: Verify the Installation
Run the following command:
python -c "import streamlit, cv2, numpy, PIL, ultralytics; print('Dependencies imported successfully')"


If the imports succeed, the main dependencies are available.
8. Configure the Model Path
Before running the application, update the load_model() function in newapp.py.
The application currently searches for:
runs/detect/train3/weights/best.pt


However, the supplied model is located at:
best.pt


One solution is to update the function to load the model from the project root:
def load_model():    model_path = os.path.join(".", "best.pt")    if not os.path.exists(model_path):        st.error(f"Error: Model file not found at {model_path}")        return None    return YOLO(model_path)



To inspect the model's supported class labels, run:
python -c "from ultralytics import YOLO; m=YOLO('best.pt'); print(m.names)"


This helps verify which classes the model can actually detect.
9. Running the Application
Navigate to the project directory and execute:
streamlit run newapp.py


Streamlit will display a local URL, usually:
http://localhost:8501


Open the URL in your browser to access the application.
Image Recognition Workflow
1. Select Image Recognition from the sidebar.
2. Choose whether to display an enhanced image.
3. Upload a JPG, JPEG, or PNG image.
4. Review the image and detection results.
5. Inspect the predicted class labels and confidence scores.
Implementation note: In the supplied code, image enhancement is applied during inference even when the enhancement checkbox is unchecked. The checkbox currently controls the enhanced-image preview rather than the preprocessing used for prediction.
Video Recognition Workflow
1. Select Video Recognition.
2. Upload an MP4, AVI, or MOV file.
3. Preview the video.
4. Select the video-processing option.
5. Wait for frame processing to complete.
6. Inspect the annotated frames and class-level summary.
Implementation note: The video enhancement checkbox is not consistently connected to the video-processing function, so frames are enhanced regardless of the checkbox setting.
Live Detection Workflow
1. Select Live Detection.
2. Start webcam detection.
3. Grant camera permission if requested.
4. View live predictions and tracking IDs.
5. Stop detection using the provided control.
Webcam stopping behaviour may require further adjustment because Streamlit reruns the application when interface interactions occur. Test the webcam workflow in the intended deployment environment.
10. Model Training and Fine-Tuning
The tra.py script defines a YOLOv8 training workflow.
It uses a pretrained yolov8n.pt checkpoint, trains the model on a configured dataset, validates the resulting model, and exports it to ONNX format.
The script references:
input/data.yaml


This file must be created and configured for your dataset before training.
Dataset Configuration
A typical YOLO object-detection dataset contains:
- Training images.
- Validation images.
- Optional test images.
- Corresponding annotation files.
- A YAML file describing the dataset and class names.
An example configuration is shown below. Replace the placeholder paths and names with your actual dataset values.
path: ./input/dataset

train: images/train
val: images/val
test: images/test

names:
  0: class_one
  1: class_two


The class names shown above are placeholders and do not represent verified classes in the supplied model.
Running Training
Once the dataset and configuration are available, run:
python tra.py


The supplied training script requests 100 epochs, an image size of 640, a batch size of 16, data augmentation, checkpoint saving, validation, and CPU execution.
Training time depends on the dataset size and available hardware. CPU training may be slow, and caching images in memory can consume substantial RAM.
Ultralytics saves training outputs in a run directory under runs/detect/. The exact directory name may change between runs. Update the application to load the intended checkpoint.
11. Detection Parameters
The supplied application uses the following main settings:
- Confidence threshold: 0.80. Detections below this threshold are generally filtered out. A high threshold may reduce false positives but can also miss valid detections.
- Model IoU threshold: 0.45. Used by non-maximum suppression to manage overlapping bounding boxes.
- Tracker IoU threshold: 0.30. Used to associate detections with existing tracks.
- Maximum missed tracking updates: 10. Determines how long an unmatched track can remain active.
- Video sampling rate: approximately 1 FPS. Reduces the number of frames sent to the model.
- Recurring video detection filter: at least 3 sampled observations. Used to retain detections that appear repeatedly in the video summary.
These values are currently hard-coded rather than exposed as adjustable interface controls.
12. Output and Results
Depending on the selected mode, the application can display:
- Original images and annotated predictions.
- Bounding boxes around detected objects.
- Predicted class labels.
- Confidence scores.
- Annotated video frames.
- Tracking IDs for video and webcam detection.
- A summary of detections by class.
- Average confidence scores for selected video detections.
A confidence score is a model-generated score, not necessarily a calibrated probability that the predicted disease is present.
Detection quality depends on image quality, lighting, camera angle, training data, annotation quality, and similarity between the input images and the training dataset.
Actual performance should be measured using an independent labelled test set.
13. Troubleshooting
Model File Not Found
Ensure that best.pt exists and that the application loads it from the correct location.
Missing Python Packages
Activate the virtual environment and install the required dependencies.
Webcam Not Working
- Check operating-system camera permissions.
- Close other applications using the webcam.
- Verify that the webcam is connected.
- Test locally if the application is running on a remote server.
Video Cannot Be Opened
Try an MP4, AVI, or MOV file encoded with a codec supported by your OpenCV installation.
Dataset Configuration Missing
Create input/data.yaml and ensure the referenced image and annotation folders exist.
Training Is Slow or Runs Out of Memory
- Reduce the batch size.
- Disable caching if RAM is limited.
- Use a compatible GPU where available.
- Consider reducing image size while evaluating its impact on model performance.
No Objects Are Detected
The confidence threshold is set to 0.80. Evaluate alternative thresholds on a labelled validation dataset and compare precision and recall before changing the application settings.
Image Enhancement Appears Inconsistent
Review the preprocessing functions and connect the enhancement checkbox to the actual inference pipeline if users should be able to enable or disable enhancement.
14. Limitations and Future Improvements
Potential improvements include:
- Adding a requirements.txt file with tested dependency versions.
- Documenting the dataset and its class labels.
- Making the model path configurable.
- Adding adjustable confidence and IoU thresholds.
- Correctly connecting image enhancement controls to inference.
- Improving video-processing and webcam error handling.
- Exporting detection results to CSV or JSON.
- Reporting precision, recall, mAP50, and mAP50-95.
- Evaluating performance on independent test data.
- Using a dedicated tracking algorithm such as ByteTrack where appropriate.
- Adding automated tests for preprocessing, inference, and result generation.
- Packaging the application for reproducible deployment.
- Evaluating the model under different lighting, camera angles, coconut varieties, and symptom stages.
15. Applications
Potential application areas include:
- Preliminary visual screening of coconut crops.
- Agricultural research and experimentation.
- Computer vision and deep-learning education.
- Prototype systems for plantation monitoring.
- Development of agricultural decision-support applications.
These are potential applications rather than validated production capabilities.
16. Disclaimer
This project is an experimental computer vision application. It does not provide a confirmed agricultural diagnosis and should not be used as the sole basis for pesticide application, treatment, or crop-management decisions. Predictions should be verified by qualified agricultural professionals.
17. Conclusion
The Coconut Disease Detection project demonstrates how YOLOv8, OpenCV, and Streamlit can be combined to build an interactive object-detection application for agricultural imagery.
By supporting images, videos, and webcam input, the application provides multiple ways to inspect and visualize model predictions. Further dataset validation, model evaluation, interface improvements, and deployment testing can help make the system more reliable and practical.

Technologies: Python | YOLOv8 | Ultralytics | OpenCV | Streamlit | PyTorch | NumPy
