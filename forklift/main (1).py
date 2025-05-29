import cv2
from ultralytics import YOLO
import time

def run_webcam_detection():
    # Load the model
    model = YOLO('best.pt')  # Path to your trained model weights
    
    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam, change if using a different camera
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set webcam properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Webcam detection started. Press 'q' to quit.")
    
    # FPS calculation variables
    prev_time = time.time()
    fps = 0
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Calculate FPS
        current_time = time.time()
        if current_time - prev_time >= 1:
            fps = int(1 / (current_time - prev_time))
            prev_time = current_time
        
        # Run inference
        results = model(frame, conf=0.25)  # Adjust confidence threshold as needed
        
        # Visualize results on frame
        annotated_frame = results[0].plot()
        
        # Add FPS counter to the frame
        cv2.putText(annotated_frame, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_detection()