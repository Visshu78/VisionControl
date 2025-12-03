import cv2
import mediapipe as mp
import time

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Check if the video capture is opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize MediaPipe Hands
mphands = mp.solutions.hands
hands = mphands.Hands()
mpDraw = mp.solutions.drawing_utils

# Initialize FPS variables
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    
    # Check if the frame is captured successfully
    if not success:
        print("Failed to grab frame")
        break
    
    # Convert the frame to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process the frame to find hands
    results = hands.process(imgRGB)
    
    # Check if any hand landmarks are detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # Get the height, width, and channel of the frame
                h, w, c = img.shape
                # Calculate the pixel coordinates of the landmarks
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"Landmark ID: {id}, Coordinates: ({cx}, {cy})")

                # Draw a circle on the first landmark (usually the wrist)
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            # Draw the hand landmarks
            mpDraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

    # Calculate Frames Per Second (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    # Display FPS on the frame
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # Show the frame with landmarks
    cv2.imshow("Hand Tracking", img)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
