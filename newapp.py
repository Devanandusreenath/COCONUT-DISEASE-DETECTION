import os
import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
from tempfile import NamedTemporaryFile
from collections import defaultdict

def load_model():
    # Fixed model path - no custom selection
    model_path = os.path.join('.', 'runs', 'detect', 'train3', 'weights', 'best.pt')
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}")
        return None
    return YOLO(model_path)

def enhance_image(image):
    """Apply image enhancement techniques to improve detection"""
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Apply contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Apply slight Gaussian blur to reduce noise
    enhanced_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
    
    return enhanced_image

def predict_image(image, model):
    # Fixed confidence and IOU thresholds
    conf_threshold = 0.80
    iou_threshold = 0.45
    
    # Enhance image before detection
    processed_image = enhance_image(image)
    
    # Run prediction
    results = model(processed_image, conf=conf_threshold, iou=iou_threshold)[0]
    
    # Convert to numpy array if it's still a PIL image
    if isinstance(image, Image.Image):
        display_image = np.array(image)
    else:
        display_image = image.copy()
    
    # Draw predictions
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > conf_threshold:
            # Use different colors for different classes
            class_id = int(class_id)
            color = get_color_for_class(class_id)
            
            cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{results.names[class_id].upper()} {score:.2f}"
            
            # Add better text display with background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_image, 
                          (int(x1), max(0, int(y1) - text_size[1] - 10)), 
                          (int(x1) + text_size[0], int(y1)), 
                          color, -1)
            cv2.putText(display_image, label, (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return display_image

def get_color_for_class(class_id):
    """Return a distinct color for each class ID"""
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    return colors[class_id % len(colors)]

def iou(box1, box2):
    """Calculate Intersection over Union for two boxes"""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

class ObjectTracker:
    """Simple tracker to maintain temporal consistency"""
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0
        self.max_disappeared = 10
        self.min_confidence = 0.80  # Updated to fixed value
    
    def update(self, detections):
        """Update tracked objects with new detections"""
        # Format: (x1, y1, x2, y2, class_id, confidence)
        current_objects = {}
        
        # No objects currently being tracked
        if len(self.tracked_objects) == 0:
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, class_id, confidence = detection
                if confidence >= self.min_confidence:
                    self.tracked_objects[self.next_id] = {
                        "box": (x1, y1, x2, y2),
                        "class_id": class_id,
                        "confidence": confidence,
                        "disappeared": 0,
                    }
                    current_objects[self.next_id] = self.tracked_objects[self.next_id]
                    self.next_id += 1
            return current_objects
        
        # We have existing tracked objects, so match new detections
        if len(detections) > 0:
            # For each detection, find the best matching tracked object
            used_tracked_ids = set()
            for detection in detections:
                x1, y1, x2, y2, class_id, confidence = detection
                if confidence < self.min_confidence:
                    continue
                
                best_iou_val = 0
                best_tracked_id = None
                
                for tracked_id, tracked_obj in self.tracked_objects.items():
                    if tracked_id in used_tracked_ids or tracked_obj["class_id"] != class_id:
                        continue
                        
                    curr_iou = iou(tracked_obj["box"], (x1, y1, x2, y2))
                    if curr_iou > best_iou_val:
                        best_iou_val = curr_iou
                        best_tracked_id = tracked_id
                
                # If we found a match, update it
                if best_tracked_id is not None and best_iou_val > 0.3:
                    # Update with a weighted average to smooth tracking
                    alpha = 0.7  # Weight for the new detection
                    old_box = self.tracked_objects[best_tracked_id]["box"]
                    new_box = (
                        int(alpha * x1 + (1-alpha) * old_box[0]),
                        int(alpha * y1 + (1-alpha) * old_box[1]),
                        int(alpha * x2 + (1-alpha) * old_box[2]),
                        int(alpha * y2 + (1-alpha) * old_box[3])
                    )
                    
                    # Smooth confidence as well
                    old_conf = self.tracked_objects[best_tracked_id]["confidence"]
                    new_conf = alpha * confidence + (1-alpha) * old_conf
                    
                    self.tracked_objects[best_tracked_id]["box"] = new_box
                    self.tracked_objects[best_tracked_id]["confidence"] = new_conf
                    self.tracked_objects[best_tracked_id]["disappeared"] = 0
                    used_tracked_ids.add(best_tracked_id)
                    current_objects[best_tracked_id] = self.tracked_objects[best_tracked_id]
                else:
                    # No match, register as a new object
                    self.tracked_objects[self.next_id] = {
                        "box": (x1, y1, x2, y2),
                        "class_id": class_id,
                        "confidence": confidence,
                        "disappeared": 0,
                    }
                    current_objects[self.next_id] = self.tracked_objects[self.next_id]
                    self.next_id += 1
            
            # Update disappearance counts for unmatched tracks
            for tracked_id in self.tracked_objects:
                if tracked_id not in used_tracked_ids:
                    self.tracked_objects[tracked_id]["disappeared"] += 1
                    
                    # Keep track if it hasn't been gone too long
                    if self.tracked_objects[tracked_id]["disappeared"] < self.max_disappeared:
                        current_objects[tracked_id] = self.tracked_objects[tracked_id]
            
            # Remove tracks that have disappeared for too long
            self.tracked_objects = {k: v for k, v in self.tracked_objects.items() 
                                   if v["disappeared"] < self.max_disappeared}
        
        return current_objects

def process_video_live(video_path, model):
    # Fixed confidence and IOU thresholds
    conf_threshold = 0.80
    iou_threshold = 0.45
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Unable to open the video file.")
        return [], {}
    
    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    detected_frames = []
    seen_objects = defaultdict(int)
    tracker = ObjectTracker()
    frame_id = 0
    
    # Use configuration values from params instead of creating new sliders
    with st.expander("Detection Progress", expanded=True):
        show_live = st.checkbox("Show Live Preview", value=True, key="video_show_live")
    
    # Process one frame per second
    frames_to_skip = max(1, fps - 1)  # Skip frames to achieve 1 FPS processing
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        progress = min(1.0, frame_id / total_frames) if total_frames > 0 else 0
        progress_bar.progress(progress)
        status_text.text(f"Processing: {frame_id}/{total_frames} frames")
        
        # Only process 1 frame per second (based on video FPS)
        if frame_id % fps == 0:
            # Enhance image before detection
            enhanced_frame = enhance_image(frame)
            
            # Run YOLOv8 detection with fixed thresholds
            results = model(enhanced_frame, conf=conf_threshold, iou=iou_threshold)[0]
            
            # Format detections for tracker
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                score = float(box.conf[0])
                class_id = int(box.cls[0])
                detections.append((x1, y1, x2, y2, class_id, score))
            
            # Update tracker
            tracked_objects = tracker.update(detections)
            
            # Copy the frame for display
            display_frame = frame.copy()
            frame_has_detections = False
            
            # Draw all current tracked objects
            for obj_id, obj_data in tracked_objects.items():
                x1, y1, x2, y2 = obj_data["box"]
                class_id = obj_data["class_id"]
                score = obj_data["confidence"]
                
                color = get_color_for_class(class_id)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                
                # Add tracking ID to the label
                label = f"{model.names[class_id]} #{obj_id} {score:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                cv2.rectangle(display_frame, 
                              (x1, max(0, y1 - text_size[1] - 5)), 
                              (x1 + text_size[0], y1), 
                              color, -1)
                cv2.putText(display_frame, label, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Mark as having detections
                frame_has_detections = True
                
                # Track unique objects
                obj_key = (class_id, obj_id)
                if obj_key not in seen_objects:
                    seen_objects[obj_key] = {
                        "class_id": class_id,
                        "confidence": score,
                        "first_seen": frame_id,
                        "last_seen": frame_id,
                        "frames_visible": 1
                    }
                else:
                    seen_objects[obj_key]["last_seen"] = frame_id
                    seen_objects[obj_key]["frames_visible"] += 1
            
            # Save frames with detections
            if frame_has_detections:
                detected_frames.append(display_frame.copy())
            
            # Display frame
            if show_live:
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                display_size = (640, 360)  # Resize for smoother display
                display_frame_resized = cv2.resize(frame_rgb, display_size)
                stframe.image(display_frame_resized, caption="Live Detection", channels="RGB")
            
            # Skip frames to achieve 1 FPS processing
            for _ in range(frames_to_skip):
                cap.read()  # Read and discard frames
                frame_id += 1
        
        frame_id += 1
    
    cap.release()
    progress_bar.progress(1.0)
    status_text.text(f"Processing complete: Analyzed at 1 frame per second")
    
    # Filter seen objects to only include significant detections
    filtered_objects = {}
    for obj_key, obj_data in seen_objects.items():
        if obj_data["frames_visible"] >= 3:  # Object must be visible for at least 3 frames (3 seconds)
            filtered_objects[obj_key] = obj_data
    
    return detected_frames, filtered_objects

def live_webcam_detection(model):
    # Fixed confidence and IOU thresholds
    conf_threshold = 0.80
    iou_threshold = 0.45
    
    # Initialize session state for stop button if not exists
    if 'stop_webcam' not in st.session_state:
        st.session_state.stop_webcam = False
    
    # Create stop button that sets the state
    if st.button("Stop Webcam Detection", key="stop_button"):
        st.session_state.stop_webcam = True
        return
    
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        st.error("Error: Unable to access the webcam.")
        return
    
    # Placeholder for displaying frames
    stframe = st.empty()
    
    # Simplified settings - just show/hide enhancement option
    with st.expander("Detection Controls", expanded=True):
        enable_enhancement = st.checkbox("Enable Image Enhancement", value=True, key="webcam_enhancement")
    
    tracker = ObjectTracker()
    frame_counter = 0
    detections_log = []
    
    # Get webcam FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default to 30 FPS if not available
    
    while cap.isOpened() and not st.session_state.stop_webcam:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        
        # Process every frame instead of once per second
        # Apply image enhancement if enabled
        if enable_enhancement:
            processed_frame = enhance_image(frame)
        else:
            processed_frame = frame
        
        # Run YOLOv8 detection with fixed thresholds
        results = model(processed_frame, conf=conf_threshold, iou=iou_threshold)[0]
        
        # Format detections for tracker
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            class_id = int(box.cls[0])
            detections.append((x1, y1, x2, y2, class_id, score))
        
        # Update tracker
        tracked_objects = tracker.update(detections)
        
        # Log detections for later analysis
        if detections:
            detections_log.append({
                "frame": frame_counter,
                "detections": [
                    {"class": model.names[d[4]], "confidence": d[5]} 
                    for d in detections if d[5] >= conf_threshold
                ]
            })
            
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Draw all current tracked objects
        for obj_id, obj_data in tracker.tracked_objects.items():
            if obj_data["disappeared"] > 0:
                continue  # Skip disappeared objects
                
            x1, y1, x2, y2 = obj_data["box"]
            class_id = obj_data["class_id"]
            score = obj_data["confidence"]
            
            color = get_color_for_class(class_id)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add tracking ID to label
            label = f"{model.names[class_id]} #{obj_id} {score:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(display_frame, 
                          (x1, max(0, y1 - text_size[1] - 5)), 
                          (x1 + text_size[0], y1), 
                          color, -1)
            cv2.putText(display_frame, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add FPS information
        cv2.putText(display_frame, f"FPS: {fps:.1f} | Processing: All Frames", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        display_size = (640, 360)  # Resize for smoother display
        display_frame_resized = cv2.resize(frame_rgb, display_size)
        stframe.image(display_frame_resized, caption="Live Detection (All Frames)", channels="RGB")
        
    cap.release()
    cv2.destroyAllWindows()
    
    # Reset the stop flag for next time
    st.session_state.stop_webcam = False
    
    # Show detection summary
    if detections_log:
        st.subheader("Webcam Detection Summary")
        detection_counts = defaultdict(int)
        for entry in detections_log:
            for det in entry["detections"]:
                detection_counts[det["class"]] += 1
        
        # Display counts
        st.write("Detected objects:")
        for cls, count in detection_counts.items():
            st.write(f"- {cls}: {count}")

def show_detected_frames(detected_frames, max_frames=10):
    if not detected_frames:
        st.warning("No significant detections found in the video.")
        return
    
    st.subheader("Key Detection Frames")
    
    # Show a subset of frames if there are too many
    display_frames = detected_frames
    if len(detected_frames) > max_frames:
        step = len(detected_frames) // max_frames
        display_frames = detected_frames[::step]
        st.info(f"Showing {len(display_frames)} key frames out of {len(detected_frames)} detected frames.")
    
    # Use columns to display multiple images per row
    cols = 2
    for i in range(0, len(display_frames), cols):
        row_frames = display_frames[i:min(i+cols, len(display_frames))]
        row = st.columns(cols)
        
        for j, frame in enumerate(row_frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            row[j].image(frame_rgb, use_column_width=True)
            row[j].caption(f"Frame #{i+j+1}")

def show_summary(seen_objects, model):
    if not seen_objects:
        st.warning("No significant objects detected in the video.")
        return
    
    st.subheader("Detection Summary")
    
    # Count by class
    class_counts = defaultdict(int)
    class_confidence = defaultdict(list)
    
    for obj_key, obj_data in seen_objects.items():
        class_id = obj_data["class_id"]
        class_name = model.names[class_id]
        class_counts[class_name] += 1
        class_confidence[class_name].append(obj_data["confidence"])
    
    # Display summary in a nice table
    data = []
    for category, count in class_counts.items():
        avg_conf = sum(class_confidence[category]) / len(class_confidence[category])
        data.append({
            "Category": category,
            "Count": count,
            "Avg. Confidence": f"{avg_conf:.2f}"
        })
    
    st.table(data)
    
    # Visualization
    if len(class_counts) > 1:
        chart_data = {
            "Category": list(class_counts.keys()),
            "Count": list(class_counts.values())
        }
        st.bar_chart(chart_data, x="Category", y="Count")

def main():
    st.set_page_config(page_title="Enhanced Coconut Disease Detection", layout="wide")
    
    st.markdown("""
    <h1 style='text-align: center;'>Enhanced Coconut Disease Detection</h1>
    <p style='text-align: center;'>Detect diseases and pests in coconut trees with improved accuracy (1 FPS processing)</p>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("Application Settings")
    app_mode = st.sidebar.selectbox('Choose Mode:', ['Image Recognition', 'Video Recognition', 'Live Detection'], key="app_mode")
    
    # Load model - fixed to train3
    model = load_model()
    if not model:
        st.error("Failed to load model. Please check if the model file exists.")
        return
    
    st.sidebar.info(f"Model classes: {', '.join(model.names.values())}")
    
    # Display based on selected mode
    if app_mode == 'Image Recognition':
        with st.expander("Detection Settings", expanded=True):
            enable_enhancement = st.checkbox("Enable Image Enhancement", value=True, key="img_enhance")
        
        st.subheader("Upload an Image")
        uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"], key="image_upload")
        
        if uploaded_image:
            # Display original image
            image = Image.open(uploaded_image)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Original Image", use_column_width=True)
            
            with col2:
                st.subheader("Processed Image")
                if enable_enhancement:
                    enhanced_image = enhance_image(np.array(image))
                    st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
            
            # Process and display results
            st.subheader("Detection Results")
            with st.spinner("Running detection..."):
                processed_image = predict_image(image, model)
                st.image(processed_image, caption="Detection Results", use_column_width=True)
    
    elif app_mode == 'Video Recognition':
        st.subheader("Upload a Video")
        
        # Simplified settings - just show/hide enhancement
        with st.expander("Detection Settings", expanded=True):
            video_enable_enhancement = st.checkbox("Enable Image Enhancement", value=True, key="video_enhance")
        
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"], key="video_upload")
        
        if uploaded_video:
            # Save uploaded video to temp file
            temp_input = NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(uploaded_video.read())
            temp_input.close()
            
            st.video(temp_input.name)
            
            # Process video button
            if st.button("Process Video (1 FPS)", key="process_video"):
                with st.spinner("Processing video at 1 frame per second..."):
                    detected_frames, seen_objects = process_video_live(
                        temp_input.name, model
                    )
                
                # Show results if we have detections
                if detected_frames:
                    show_detected_frames(detected_frames)
                    
                    if seen_objects:
                        show_summary(seen_objects, model)
                    else:
                        st.warning("No significant objects were detected in the video.")
                else:
                    st.warning("No detections found in the video.")
    
    elif app_mode == 'Live Detection':
        st.subheader("Webcam Detection")
        st.warning("This mode requires access to your webcam.")
        
        # Display fixed settings information
        st.info("Using fixed settings: Confidence Threshold = 0.80, IOU Threshold = 0.45")
        
        if st.button("Start Webcam Detection (All Frames)", key="start_webcam"):
            live_webcam_detection(model)

if __name__ == "__main__":
    main()