import cv2
import mediapipe as mp
import pyautogui

def main():
    cap = cv2.VideoCapture(0)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    screen_width, screen_height = pyautogui.size()
    
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
            
            # Get left and right shoulder positions
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            
            # Calculate shoulder width
            shoulder_width = right_shoulder - left_shoulder
            
            # Calculate mouse cursor movement based on shoulder width
            cursor_speed = 10  # Adjust this value for sensitivity
            cursor_position = pyautogui.position()
            
            new_cursor_x = cursor_position[0] + int(shoulder_width * cursor_speed)
            
            # Limit cursor position to screen width
            new_cursor_x = max(0, min(screen_width, new_cursor_x))
            
            pyautogui.moveTo(new_cursor_x, cursor_position[1])
        
        cv2.imshow('Body Movement Control', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
