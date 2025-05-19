import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def angle_between(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Load video file
cap = cv2.VideoCapture('video/WhatsApp Video 2025-05-15 at 10.56.44_a0d134bc.mp4')

# Previous positions and speeds for head and knee
prev_head_y = None
prev_head_speed = 0
prev_knee_y = None
prev_knee_speed = 0

# For smoothing speed (using small history)
speed_history_length = 3
head_speed_history = []
knee_speed_history = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        def get_point(landmark_id):
            lm = landmarks[landmark_id]
            return int(lm.x * w), int(lm.y * h)

        head = get_point(mp_pose.PoseLandmark.NOSE)
        left_knee = get_point(mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = get_point(mp_pose.PoseLandmark.RIGHT_KNEE)
        left_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_point(mp_pose.PoseLandmark.RIGHT_HIP)

        stomach = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)

        # Static pose features
        head_y = head[1]
        stomach_y = stomach[1]
        head_to_hip_diff = stomach_y - head_y

        avg_knee = ((left_knee[0] + right_knee[0]) // 2, (left_knee[1] + right_knee[1]) // 2)
        knee_y = avg_knee[1]

        body_angle = angle_between(head, stomach, avg_knee)

        # Calculate head vertical speed and acceleration
        if prev_head_y is not None:
            head_speed = prev_head_y - head_y
        else:
            head_speed = 0
        head_speed_history.append(head_speed)
        if len(head_speed_history) > speed_history_length:
            head_speed_history.pop(0)
        smooth_head_speed = sum(head_speed_history) / len(head_speed_history)

        head_acceleration = smooth_head_speed - prev_head_speed

        prev_head_y = head_y
        prev_head_speed = smooth_head_speed

        # Calculate knee vertical speed and acceleration
        if prev_knee_y is not None:
            knee_speed = prev_knee_y - knee_y
        else:
            knee_speed = 0
        knee_speed_history.append(knee_speed)
        if len(knee_speed_history) > speed_history_length:
            knee_speed_history.pop(0)
        smooth_knee_speed = sum(knee_speed_history) / len(knee_speed_history)

        knee_acceleration = smooth_knee_speed - prev_knee_speed

        prev_knee_y = knee_y
        prev_knee_speed = smooth_knee_speed

        # Print speeds and accelerations in terminal
        print(f"Head Speed: {smooth_head_speed:.2f}, Head Acceleration: {head_acceleration:.2f} | "
              f"Knee Speed: {smooth_knee_speed:.2f}, Knee Acceleration: {knee_acceleration:.2f}")

        # Fall detection logic (adjust thresholds as needed)
        fall_static = (head_to_hip_diff < 100 and body_angle < 100)
        fall_dynamic = (head_acceleration > 2 and smooth_head_speed > 1) or \
                       (knee_acceleration > 2 and smooth_knee_speed > 1)

        if fall_static or fall_dynamic:
            cv2.putText(frame, "FALL DETECTED!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Draw keypoints and lines
        cv2.circle(frame, head, 10, (0, 255, 0), -1)
        cv2.circle(frame, stomach, 10, (255, 0, 0), -1)
        cv2.circle(frame, left_knee, 10, (0, 0, 255), -1)
        cv2.circle(frame, right_knee, 10, (0, 0, 255), -1)

        cv2.line(frame, head, stomach, (100, 255, 100), 2)
        cv2.line(frame, stomach, left_knee, (100, 255, 100), 2)
        cv2.line(frame, stomach, right_knee, (100, 255, 100), 2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Fall Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
