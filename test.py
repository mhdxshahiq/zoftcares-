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

# Load video file instead of webcam
cap = cv2.VideoCapture('video/VID-20250519-WA0039.mp4')  # Change to your filename

prev_head_y = None  # to store the head's y-position from the previous frame


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get pose estimation
    results = pose.process(rgb_frame)

    # Draw landmarks and custom points
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
        
        import numpy as np  # Put this at the top of the file if not already

        # === FALL DETECTION LOGIC ===
        # Helper: angle between three points
        def angle_between(p1, p2, p3):
            a = np.array(p1)
            b = np.array(p2)
            c = np.array(p3)

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle)

       # Calculate head-to-stomach vertical distance
        head_y = head[1]
        stomach_y = stomach[1]
        head_to_hip_diff = stomach_y - head_y

        # Average knee position
        avg_knee = ((left_knee[0] + right_knee[0]) // 2, (left_knee[1] + right_knee[1]) // 2)

        # Body angle (head → stomach → knees)
        body_angle = angle_between(head, stomach, avg_knee)

        # Fall condition
        if head_to_hip_diff < 50 or body_angle < 100:
            cv2.putText(frame, "FALL DETECTED!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        

        # Draw keypoints (no labels)
        cv2.circle(frame, head, 10, (0, 255, 0), -1)
        cv2.circle(frame, stomach, 10, (255, 0, 0), -1)
        cv2.circle(frame, left_knee, 10, (0, 0, 255), -1)
        cv2.circle(frame, right_knee, 10, (0, 0, 255), -1)

        # Draw connecting lines
        cv2.line(frame, head, stomach, (100, 255, 100), 2)
        cv2.line(frame, stomach, left_knee, (100, 255, 100), 2)
        cv2.line(frame, stomach, right_knee, (100, 255, 100), 2)

        # Optional: draw all pose connections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show result
    cv2.imshow('MediaPipe Pose Detection - Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
