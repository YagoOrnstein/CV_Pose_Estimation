"""
For using this code as a library
Copy the codes under the main function and copy to new py file
Import the cv2, time and PoseModule(as pm)
Than add pm at the beginning of detector.
////
For changing the video or changing tracker to the webcam change the cap in main
"""

import mediapipe as mp
import cv2
import time


class PoseDetector():

    def __init__(self, static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.static_image_mode,
                                      self.model_complexity,
                                      self.smooth_landmarks,
                                      self.enable_segmentation,
                                      self.smooth_segmentation,
                                      self.min_detection_confidence,
                                      self.min_tracking_confidence)

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
        return lmlist


def main():
    cap = cv2.VideoCapture("videoplayback.mp4")
    prev_time = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        lmlist = detector.find_position(img, draw=False)
        if(len(lmlist)) != 0:
            cv2.circle(img, (lmlist[0][1], lmlist[0][2]), 3, (0, 255, 0), cv2.FILLED)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))

        cv2.imshow("Image", img)
        cv2.waitKey(7)


if __name__ == "__main__":
    main()
