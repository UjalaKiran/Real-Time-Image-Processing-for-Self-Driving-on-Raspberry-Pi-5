import cv2
import numpy as np

def detect_humans_and_obstacles(frame):
    original_frame = frame.copy()
    height, width = frame.shape[:2]
    status = "None"

    # HUMAN DETECTION (HOG + SVM)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    resized = cv2.resize(frame, (640, 360))
    scale_x = width / resized.shape[1]
    scale_y = height / resized.shape[0]

    human_boxes, _ = hog.detectMultiScale(resized, winStride=(8, 8), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in human_boxes:
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)

        center_y = (y1 + y2) // 2
        if center_y > height * 0.6:
            status = "Near"
            color = (0, 0, 255)  # Red
        else:
            status = "Far"
            color = (0, 255, 255)  # Yellow

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Human - {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    #  OTHER OBSTACLE DETECTION
    roi = original_frame[int(height * 0.5):, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for orange and white (adjust these values as needed)
    lower_orange = np.array([0, 100, 100])
    upper_orange = np.array([20, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])

    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_or(mask_orange, mask_white)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Reduced area threshold, adjust as needed
            x, y, w, h = cv2.boundingRect(cnt)
            center_y = y + h // 2

            if center_y > roi.shape[0] * 0.6:
                obs_status = "Near"
                color = (255, 0, 0)  # Blue
            else:
                obs_status = "Far"
                color = (0, 255, 0)  # Green

            x_full = x
            y_full = y + int(height * 0.5)
            cv2.rectangle(frame, (x_full, y_full), (x_full + w, y_full + h), color, 2)
            cv2.putText(frame, f"Obstacle - {obs_status}", (x_full, y_full - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return {"status": status, "frame": frame}


if __name__ == "__main__":
    path = r"C:\Users\raoas\OneDrive\Desktop\PXL_20250325_043754655.TS.mp4"
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_humans_and_obstacles(frame)

        cv2.putText(result["frame"], f"Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Human + Obstacle Detection", result["frame"])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
