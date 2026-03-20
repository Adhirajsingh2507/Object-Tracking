import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Finger Tip Landmark IDs (According to Mediapipe)
tipIds = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)   # mirror view
        h, w, c = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        fingers = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Access all landmark positions
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    lmList.append((int(lm.x * w), int(lm.y * h)))

                # Count fingers
                if lmList:
                    # Thumb check (special case)
                    if lmList[tipIds[0]][0] < lmList[tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # Other 4 fingers
                    for id in range(1, 5):
                        if lmList[tipIds[id]][1] < lmList[tipIds[id] - 2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                # Draw Hand Landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display Finger Count
        if fingers:
            text = f"Fingers Up: {fingers.count(1)}"
            cv2.putText(frame, text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Detection", frame)

        # Press 'ESC' to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
