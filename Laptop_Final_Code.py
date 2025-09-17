import cv2
import numpy as np
import os
import time

# File paths
input_video = "D:/DIP_Labs/SemesterProject/DIP Project Videos/PXL_20250325_043754655.TS.mp4"
output_video = "D:/DIP_Labs/SemesterProject/Processing_Combined/output_combined.mp4"
output_dir = "D:/DIP_Labs/SemesterProject/output_combined_frames"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Counter for saving debug frames
debug_frame_count = 0
max_debug_frames = 5

# Store previous lane polynomials for smoothing
prev_left_poly = None
prev_right_poly = None

def region_selection(image):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    rows, cols = image.shape[:2]

    # Adjusted polygon points to include more area on the sides
    bottom_left = [cols * 0.0, rows * 0.98]
    top_left = [cols * 0.45, rows * 0.6]
    top_right = [cols * 0.55, rows * 0.6]
    bottom_right = [cols * 1.0, rows * 0.98]

    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image, vertices

def detect_lanes(frame):
    global prev_left_poly, prev_right_poly

    # Ensure frame is grayscale
    if len(frame.shape) > 2:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(
        gray,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=20
    )

    # Initialize lists for left and right lines
    left_lines = []
    right_lines = []

    # Original frame for annotation
    annotated_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate slope
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                # Classify lines based on slope
                if -1.0 < slope < -0.3:  # Left lane (negative slope)
                    left_lines.append((x1, y1, x2, y2))
                elif 0.3 < slope < 1.0:  # Right lane (positive slope)
                    right_lines.append((x1, y1, x2, y2))

    # Average left and right lines
    def average_lines(lines):
        if not lines:
            return None
        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        if len(x_coords) == 0:
            return None
        poly = np.polyfit(x_coords, y_coords, 2)  # Quadratic fit
        return poly

    left_poly = average_lines(left_lines)
    right_poly = average_lines(right_lines)

    # Update previous polynomials for smoothing
    if left_poly is not None:
        prev_left_poly = left_poly
    if right_poly is not None:
        prev_right_poly = right_poly

    # Draw lane lines
    height, width = gray.shape
    lane_points = []

    def draw_poly(poly, color):
        if poly is None:
            return
        y = np.linspace(0, height - 1, 100)
        x = poly[0] * y**2 + poly[1] * y + poly[2]
        points = np.array([(int(x[i]), int(y[i])) for i in range(len(x)) if 0 <= x[i] < width])
        if len(points) > 1:
            cv2.polylines(annotated_frame, [points], False, color, 2)
            lane_points.append(points[-1])  # Bottom point for drift calculation

    draw_poly(left_poly, (0, 255, 0))  # Green for left lane
    draw_poly(right_poly, (0, 0, 255))  # Red for right lane

    # Determine vehicle drift
    drift = "Centered"
    if len(lane_points) == 2:  # Both lanes detected
        lane_center = (lane_points[0][0] + lane_points[1][0]) // 2
        frame_center = width // 2
        if lane_center < frame_center - 50:
            drift = "Right drift"
        elif lane_center > frame_center + 50:
            drift = "Left drift"

    # Output dictionary
    output_dict = {
        "left_lane_poly": left_poly.tolist() if left_poly is not None else None,
        "right_lane_poly": right_poly.tolist() if right_poly is not None else None,
        "drift": drift,
        "left_poly": left_poly,
        "right_poly": right_poly
    }

    return output_dict

def detect_humans_and_obstacles(frame):
    original_frame = frame.copy()
    height, width = frame.shape[:2]
    status = "None"
    left_obstacles_near = []
    left_obstacles_far = []
    right_obstacles_near = []
    right_obstacles_far = []
    central_obstacles_near = []
    central_obstacles_far = []
    frame_center = width // 2
    central_left = width * 0.4
    central_right = width * 0.6

    # HUMAN DETECTION (HOG + SVM)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Frame is already downsampled in process_video (scale_factor = 0.75)
    # HOG expects 640x360, so resize accordingly if needed
    target_width, target_height = 640, 360
    scale_x = width / target_width
    scale_y = height / target_height
    resized = cv2.resize(frame, (target_width, target_height))

    human_boxes, _ = hog.detectMultiScale(resized, winStride=(8, 8), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in human_boxes:
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        x2 = int((x + w) * scale_x)
        y2 = int((y + h) * scale_y)
        w_full = x2 - x1
        h_full = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if center_y > height * 0.6:
            obs_status = "Near"
            color = (0, 0, 255)  # Red
            status = "Near"
        else:
            obs_status = "Far"
            color = (0, 255, 255)  # Yellow

        # Classify by horizontal position
        if center_x < central_left:
            if obs_status == "Near":
                left_obstacles_near.append((x1, y1, w_full, h_full, center_x))
            else:
                left_obstacles_far.append((x1, y1, w_full, h_full, center_x))
        elif center_x > central_right:
            if obs_status == "Near":
                right_obstacles_near.append((x1, y1, w_full, h_full, center_x))
            else:
                right_obstacles_far.append((x1, y1, w_full, h_full, center_x))
        else:
            if obs_status == "Near":
                central_obstacles_near.append((x1, y1, w_full, h_full, center_x))
            else:
                central_obstacles_far.append((x1, y1, w_full, h_full, center_x))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Human - {obs_status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # OTHER OBSTACLE DETECTION
    roi = original_frame[int(height * 0.5):, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for orange and white
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
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w // 2
            center_y = y + h // 2
            y_full = y + int(height * 0.5)
            center_y_full = center_y + int(height * 0.5)

            if center_y > roi.shape[0] * 0.6:
                obs_status = "Near"
                color = (255, 0, 0)  # Blue
                status = "Near"
            else:
                obs_status = "Far"
                color = (0, 255, 0)  # Green

            # Classify by horizontal position
            if center_x < central_left:
                if obs_status == "Near":
                    left_obstacles_near.append((x, y_full, w, h, center_x))
                else:
                    left_obstacles_far.append((x, y_full, w, h, center_x))
            elif center_x > central_right:
                if obs_status == "Near":
                    right_obstacles_near.append((x, y_full, w, h, center_x))
                else:
                    right_obstacles_far.append((x, y_full, w, h, center_x))
            else:
                if obs_status == "Near":
                    central_obstacles_near.append((x, y_full, w, h, center_x))
                else:
                    central_obstacles_far.append((x, y_full, w, h, center_x))

            cv2.rectangle(frame, (x, y_full), (x + w, y_full + h), color, 2)
            cv2.putText(frame, f"Obstacle - {obs_status}", (x, y_full - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return {
        "status": status,
        "frame": frame,
        "left_obstacles_near": left_obstacles_near,
        "left_obstacles_far": left_obstacles_far,
        "right_obstacles_near": right_obstacles_near,
        "right_obstacles_far": right_obstacles_far,
        "central_obstacles_near": central_obstacles_near,
        "central_obstacles_far": central_obstacles_far
    }

def make_decision(lane_info, obstacle_info, roi_vertices):
    # Size threshold for considering obstacles as significant (in pixels)
    size_threshold = 1000  # Area (w * h) below this is considered too small/far

    # Function to check if a point is within the ROI
    def point_in_roi(point):
        x, y = point
        return cv2.pointPolygonTest(roi_vertices, (x, y), False) >= 0

    # Check for significant obstacles within the ROI
    central_near_in_roi = False
    central_far_in_roi = False
    left_near_in_roi = False
    right_near_in_roi = False

    # Check central obstacles
    for obs in obstacle_info["central_obstacles_near"]:
        x, y, w, h, center_x = obs
        center_y = y + h // 2
        if point_in_roi((center_x, center_y)) and w * h >= size_threshold:
            central_near_in_roi = True
            break

    for obs in obstacle_info["central_obstacles_far"]:
        x, y, w, h, center_x = obs
        center_y = y + h // 2
        if point_in_roi((center_x, center_y)) and w * h >= size_threshold:
            central_far_in_roi = True
            break

    # Check left and right obstacles within ROI
    for obs in obstacle_info["left_obstacles_near"]:
        x, y, w, h, center_x = obs
        center_y = y + h // 2
        if point_in_roi((center_x, center_y)) and w * h >= size_threshold:
            left_near_in_roi = True
            break

    for obs in obstacle_info["right_obstacles_near"]:
        x, y, w, h, center_x = obs
        center_y = y + h // 2
        if point_in_roi((center_x, center_y)) and w * h >= size_threshold:
            right_near_in_roi = True
            break

    # Decision-making based on obstacle position and proximity
    if central_near_in_roi:
        return "Stop"  # Obstacle is near and in the middle (central region)

    if left_near_in_roi and not right_near_in_roi:
        return "Turn Right"  # Near obstacle on left, no near obstacle on right

    if right_near_in_roi and not left_near_in_roi:
        return "Turn Left"  # Near obstacle on right, no near obstacle on left

    if central_far_in_roi and not central_near_in_roi:
        return "Move Forward"  # Obstacle is far in the ROI

    # If no significant obstacles in ROI, use lane detection for drift correction
    road_detected = lane_info["left_poly"] is not None and lane_info["right_poly"] is not None
    if road_detected:
        drift = lane_info["drift"]
        if drift == "Left drift" and not right_near_in_roi:
            return "Turn Right"
        elif drift == "Right drift" and not left_near_in_roi:
            return "Turn Left"
        else:
            return "Move Forward"

    return "Move Forward"  # Default if no obstacles or lanes detected

def process_video(inp_vid, out_vid):
    if not os.path.exists(inp_vid):
        print(f"Error: Input video {inp_vid} not found")
        return
    cap = cv2.VideoCapture(inp_vid)
    if not cap.isOpened():
        print(f"Error: Could not open video file {inp_vid}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video resolution: {width}x{height}, FPS: {fps}, Total Frames: {frame_count}")

    # Calculate target delay for real-time playback, then slow it down
    target_delay_ms = int(1000 / fps)  # e.g., ~33ms for 30 FPS
    slowdown_factor = 2  # 2x slower (e.g., 66ms per frame)
    target_delay_ms *= slowdown_factor

    # Downsample frame for faster processing
    scale_factor = 0.75  # Reduce resolution by 25%
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_vid, fourcc, fps, (width, height))

    frame_idx = 0
    start_time = time.time()
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Downsample frame
        frame = cv2.resize(frame, (new_width, new_height))

        # Step 1: Apply ROI
        roi_frame, roi_vertices = region_selection(frame)
        # Step 2: Lane detection (process on ROI, annotate on original frame)
        lane_info = detect_lanes(roi_frame)
        annotated_frame = frame.copy()

        def draw_poly(poly, color):
            if poly is None:
                return
            y = np.linspace(0, new_height - 1, 100)
            x = poly[0] * y ** 2 + poly[1] * y + poly[2]
            points = np.array([(int(x[i]), int(y[i])) for i in range(len(x)) if 0 <= x[i] < new_width])
            if len(points) > 1:
                cv2.polylines(annotated_frame, [points], False, color, 2)

        draw_poly(lane_info["left_poly"], (0, 255, 0))
        draw_poly(lane_info["right_poly"], (0, 0, 255))
        cv2.putText(annotated_frame, f"Drift: {lane_info['drift']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        # Step 3: Obstacle detection (on original frame for full context)
        obstacle_info = detect_humans_and_obstacles(annotated_frame)
        annotated_frame = obstacle_info["frame"]
        # Step 4: Make driving decision
        decision = make_decision(lane_info, obstacle_info, roi_vertices[0])
        cv2.putText(annotated_frame, f"Decision: {decision}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Resize back to original size for display and saving
        annotated_frame = cv2.resize(annotated_frame, (width, height))

        # Display the frame
        cv2.imshow("Self-Driving Output", annotated_frame)

        # Save debug frames
        global debug_frame_count
        if debug_frame_count < max_debug_frames:
            cv2.imwrite(os.path.join(output_dir, f"debug_combined_frame_{debug_frame_count}.jpg"), annotated_frame)
            debug_frame_count += 1
        out.write(annotated_frame)

        # Calculate processing time and adjust waitKey
        processed_frames += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            current_fps = processed_frames / elapsed_time
        else:
            current_fps = 0
        if frame_idx % 30 == 0:
            print(
                f"Processed frame {frame_idx}/{frame_count}, Drift: {lane_info['drift']}, Obstacle: {obstacle_info['status']}, Decision: {decision}, FPS: {current_fps:.2f}")

        # Adjust waitKey to match slowed-down target FPS and stop on any keypress
        processing_time_ms = (time.time() - start_time) * 1000 / processed_frames if processed_frames > 0 else 0
        wait_time = max(1, int(target_delay_ms - processing_time_ms))
        key = cv2.waitKey(wait_time)
        if key != -1:  # Any keypress will exit the loop
            break

        frame_idx += 1

    # Calculate final FPS
    total_time = time.time() - start_time
    final_fps = processed_frames / total_time if total_time > 0 else 0
    print(f"Final FPS: {final_fps:.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Combined processing video saved to {out_vid}")

# Start processing the video
process_video(input_video, output_video)