import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Hand landmarker model path
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Error: hand_landmarker.task not found. Please download it first.")
    exit(1)

# Path to the downloaded model
DOWNLOADED_MODEL_PATH = '/Users/khalid/Downloads/asl_model_improved.h5'

# Load the trained model
print(f"Loading ASL model from {DOWNLOADED_MODEL_PATH}...")
model = load_model(DOWNLOADED_MODEL_PATH)
print("Model loaded!")

# Arabic letter mapping
english_to_arabic = {
    'ain': 'ع', 'al': 'ال', 'aleff': 'أ', 'bb': 'ب', 'dal': 'د', 'dha': 'ظ',
    'dhad': 'ض', 'fa': 'ف', 'gaaf': 'ق', 'ghain': 'غ', 'ha': 'هـ', 'haa': 'ح',
    'jeem': 'ج', 'kaaf': 'ك', 'khaa': 'خ', 'la': 'لا', 'laam': 'ل', 'meem': 'م',
    'nun': 'ن', 'ra': 'ر', 'saad': 'ص', 'seen': 'س', 'sheen': 'ش', 'ta': 'ط',
    'taa': 'ت', 'thaa': 'ث', 'thal': 'ذ', 'toot': 'ة', 'waw': 'و', 'ya': 'ئ',
    'yaa': 'ي', 'zay': 'ز'
}

# Create hand landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("\n=== Arabic Sign Language Recognition (Downloaded Model) ===")
print("Show your hand to the camera")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hands
    detection_result = detector.detect(mp_image)

    predicted_letter = None
    confidence = 0

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]

        # Extract landmarks
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.append([landmark.x, landmark.y, landmark.z])

        # Flatten and reshape for prediction
        landmarks_flat = np.array(landmarks).flatten().reshape(1, -1)

        # Get prediction
        pred = model.predict(landmarks_flat, verbose=0)
        max_pred = np.max(pred)
        confidence = max_pred

        if max_pred > 0.90:
            predicted_class = np.argmax(pred, axis=1)
            predicted_label = list(english_to_arabic.keys())[predicted_class[0]]
            predicted_letter = english_to_arabic[predicted_label]

        # Draw landmarks on frame
        h, w, _ = frame.shape
        for landmark in hand_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw bounding box
        x_coords = [landmark.x for landmark in hand_landmarks]
        y_coords = [landmark.y for landmark in hand_landmarks]
        xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
        ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)
        cv2.rectangle(frame, (xmin-10, ymin-10), (xmax+10, ymax+10), (0, 255, 0), 2)

    # Display prediction
    if predicted_letter:
        text = f"Letter: {predicted_letter} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"\rPredicted: {predicted_letter} (Confidence: {confidence*100:.1f}%)", end="", flush=True)
    else:
        conf_text = f"Confidence: {confidence*100:.1f}%" if confidence > 0 else "No hand detected"
        cv2.putText(frame, conf_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow('Arabic Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n\nDone!")
