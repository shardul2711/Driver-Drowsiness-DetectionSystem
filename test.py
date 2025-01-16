import cv2
from transformers import pipeline
from playsound import playsound

# Load the Hugging Face image classification model
print("Loading the model...")
pipe = pipeline("image-classification", model="chbh7051/vit-final-driver-drowsiness-detection")
print("Model loaded successfully!")

# Initialize OpenCV's video capture (use 0 for default webcam)
cap = cv2.VideoCapture(0)

# Helper function to classify a single frame
def classify_frame(frame):
    # Save the frame temporarily
    cv2.imwrite("current_frame.jpg", frame)

    # Perform classification using the pipeline
    results = pipe("current_frame.jpg")

    # Get the top label and score
    label = results[0]["label"]
    score = results[0]["score"]
    return label, score

print("Starting video capture. Press 'q' to quit.")

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize the frame for better performance (optional)
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to model's expected input size

    # Get classification results
    label, score = classify_frame(resized_frame)

    # Display results on the frame
    color = (0, 255, 0) if label == "awake" else (0, 0, 255)  # Green for awake, Red for drowsy
    text = f"{label}: {score:.2f}"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Play alert sound if drowsy is detected
    if label == "drowsy":
        playsound("beep.mp3")  # Replace "alert.mp3" with the path to your audio file

    # Show the frame in a window
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
