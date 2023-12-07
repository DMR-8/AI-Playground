import cv2
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(0)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image with the pose estimation model
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Get landmark positions
            landmarks = results.pose_landmarks.landmark
            
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Calculate angles for body orientation
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            shoulder_width = right_shoulder - left_shoulder
            
            print(f"Shoulder Width: {shoulder_width:.2f}")
        
        cv2.imshow('Body Orientation Detection', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
