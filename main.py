import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, 1, 0.7, 0.5)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

flag_width = 300
flag_height = 200
flag_speed = 5
flag_hoisting = False
waving_offset = 0
waving_speed = 0.1

# Draw Indian flag with waving effect
def draw_indian_flag(frame, x, y, width, height, time_offset):

    flag = np.zeros((height, width, 3), dtype=np.uint8)

    flag[0:height//3, :] = [0, 165, 255]
    flag[height//3:2*height//3, :] = [255, 255, 255]
    flag[2*height//3:height, :] = [0, 255, 0]

    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 8

    cv2.circle(flag, (center_x, center_y), radius, (0, 0, 255), -1)

    for i in range(24):
        angle = i * (2 * math.pi / 24)
        end_x = center_x + int(radius * math.cos(angle))
        end_y = center_y + int(radius * math.sin(angle))
        cv2.line(flag, (center_x, center_y), (end_x, end_y), (255, 255, 255), 2)

    waved_flag = np.zeros_like(flag)

    for i in range(width):
        offset = int(10 * math.sin(time_offset + i * 0.05))
        if 0 <= i + offset < width:
            waved_flag[:, i] = flag[:, i + offset]

    y = max(0, min(y, frame.shape[0] - height))
    x = max(0, min(x, frame.shape[1] - width))

    roi = frame[y:y+height, x:x+width]

    alpha = 0.9
    beta = 1.0 - alpha
    blended = cv2.addWeighted(waved_flag, alpha, roi, beta, 0)

    frame[y:y+height, x:x+width] = blended
    return frame


# Detect upward hand gesture
def detect_upward_gesture(hand_landmarks):

    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    is_middle_up = middle_finger_tip.y < middle_finger_mcp.y
    is_hand_up = middle_finger_tip.y < wrist.y

    hand_vector_x = middle_finger_tip.x - wrist.x
    hand_vector_y = middle_finger_tip.y - wrist.y
    angle = math.atan2(hand_vector_y, hand_vector_x) * 180 / math.pi

    is_vertical = -90 <= angle <= -30

    return is_middle_up and is_hand_up and is_vertical


# Main loop
prev_time = time.time()
flag_position = None
flag_max_position = 0

while True:

    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    waving_offset += waving_speed

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)

    if flag_position is None:
        flag_position = frame.shape[0]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if detect_upward_gesture(hand_landmarks):

                if not flag_hoisting:
                    flag_hoisting = True
                    flag_max_position = frame.shape[0] - flag_height - 50

                cv2.putText(frame, "Upward Gesture Detected!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if flag_hoisting:
        flag_position -= flag_speed
        if flag_position <= flag_max_position:
            flag_position = flag_max_position

    if flag_hoisting or flag_position < frame.shape[0]:
        flag_x = (frame.shape[1] - flag_width) // 2
        draw_indian_flag(frame, flag_x, int(flag_position), flag_width, flag_height, waving_offset)

    cv2.putText(frame, "Show upward hand gesture to hoist the flag", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Gesture-Controlled Flag Hoisting', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
