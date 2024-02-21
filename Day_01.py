import opencv as cv
import mediapipe as mp

# Load pre-trained hand detection model from Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Define actions corresponding to hand gestures (customize as needed)
gesture_actions = {
    "Fist": "Volume Down",
    "Five": "Volume Up",
    "Thumbs Up": "Play/Pause",
}

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark coordinates for each hand
            # Perform actions based on detected gestures (customize as needed)
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y:
                action = "Thumbs Up"
            elif hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x:
                action = "Fist"
            elif hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x:
                action = "Five"
            else:
                action = "Unknown"

            # Display the detected gesture
            cv2.putText(frame, f"Action: {gesture_actions.get(action, 'Unknown')}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the webcam feed
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
