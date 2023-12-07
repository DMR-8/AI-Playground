import cv2
import mediapipe as mp

def is_thumbs_up(hand_landmarks):
    # Thumb tip landmark ID is 4
    thumb_tip = hand_landmarks.landmark[4]
    
    # Index finger tip landmark ID is 8
    index_finger_tip = hand_landmarks.landmark[8]
    
    # Calculate the Euclidean distance between thumb tip and index finger tip
    distance = ((thumb_tip.x - index_finger_tip.x) ** 2 + 
                (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5
    
    # Determine if the distance is smaller than a threshold (gesture is recognized)
    return distance < 0.1

def main():
    cap = cv2.VideoCapture(0)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image with the hand tracking model
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Check for thumbs up gesture
                if is_thumbs_up(landmarks):
                    cv2.putText(frame, "Thumbs Up!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Hand Tracking', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
