import cv2
from transformers import pipeline
import torch
import pygame

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Load the alert sound
alert_sound = "beep.mp3"  # Replace with the path to your sound file
try:
    pygame.mixer.music.load(alert_sound)
except pygame.error as e:
    print(f"Error loading sound: {e}")
    exit()

# Load the Hugging Face image classification model for drowsiness detection
print("Loading drowsiness detection model...")
pipe = pipeline("image-classification", model="chbh7051/vit-final-driver-drowsiness-detection")
print("Drowsiness detection model loaded successfully!")

# Load YOLOv5 model for object detection
print("Loading YOLOv5 model...")
yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Replace "yolov5s" with your trained YOLOv5 model if available
print("YOLOv5 model loaded successfully!")

# Initialize OpenCV's video capture (use 0 for default webcam)
cap = cv2.VideoCapture(0)

# Helper function to classify a single frame for drowsiness
def classify_frame(frame):
    # Save the frame temporarily
    cv2.imwrite("current_frame.jpg", frame)

    # Perform classification using the pipeline
    results = pipe("current_frame.jpg")

    # Get the top label and score
    label = results[0]["label"]
    score = results[0]["score"]
    return label, score

# Helper function to detect objects using YOLOv5
def detect_objects(frame):
    results = yolo_model(frame)
    detected_objects = results.pandas().xyxy[0]  # Extract detections as a pandas DataFrame
    return detected_objects

print("Starting video capture. Press 'q' to quit.")

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize the frame for drowsiness detection (optional)
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to model's expected input size

    # Drowsiness detection
    label, score = classify_frame(resized_frame)

    # Object detection using YOLOv5
    detected_objects = detect_objects(frame)

    # Check if a phone is detected
    phone_detected = False
    for _, obj in detected_objects.iterrows():
        if obj["name"] == "cell phone":  # YOLOv5 label for a phone
            phone_detected = True
            # Draw bounding box around the phone
            x1, y1, x2, y2 = int(obj["xmin"]), int(obj["ymin"]), int(obj["xmax"]), int(obj["ymax"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for phone
            cv2.putText(frame, "Phone Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display results on the frame for drowsiness detection
    color = (0, 255, 0) if label == "awake" else (0, 0, 255)  # Green for awake, Red for drowsy
    text = f"{label}: {score:.2f}"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Play alert sound if drowsy or phone is detected
    if label == "drowsy" or phone_detected:
        if not pygame.mixer.music.get_busy():  # Check if the sound is already playing
            pygame.mixer.music.play()

    # Show the frame in a window
    cv2.imshow("Driver Drowsiness and Object Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Stop any playing sound
pygame.mixer.music.stop()
pygame.mixer.quit()
