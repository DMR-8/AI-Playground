import cv2

def main():
    cap = cv2.VideoCapture(0)
    
    # Create background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Apply thresholding to clean up the mask
        _, thresholded = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded mask
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Adjust this threshold to filter out small contours
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        
        cv2.imshow('Body Motion Detection', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
