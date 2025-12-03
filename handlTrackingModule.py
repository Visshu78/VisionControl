import cv2
import mediapipe as mp
import time

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mphands = mp.solutions.hands
        # Correct argument names for Hands initialization
        self.hands = self.mphands.Hands(self.mode, self.maxHands, 
                                        min_detection_confidence=self.detectionCon, 
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = handDetector()

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()
 
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        img = detector.findHands(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            print(lmList[4])  # Example to print the position of landmark 4

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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
