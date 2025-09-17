import cv2
import numpy as np
import os

# File paths
input_video = "output_ROI.mp4"
output_video = "output_lane_detection.mp4"
output_dir = "output_lane_frames"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Counter for saving debug frames
debug_frame_count = 0
max_debug_frames = 5

def detect_lanes(frame):

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

    # Annotate drift direction
    cv2.putText(
        annotated_frame,
        f"Drift: {drift}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Output dictionary
    output_dict = {
        "left_lane_poly": left_poly.tolist() if left_poly is not None else None,
        "right_lane_poly": right_poly.tolist() if right_poly is not None else None,
        "drift": drift
    }

    return output_dict, annotated_frame

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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_vid, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lane_info, annotated_frame = detect_lanes(frame)
        out.write(annotated_frame)

        # Save debug frames
        global debug_frame_count
        if debug_frame_count < max_debug_frames:
            cv2.imwrite(os.path.join(output_dir, f"debug_lane_frame_{debug_frame_count}.jpg"), annotated_frame)
            debug_frame_count += 1

        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}/{frame_count}, Drift: {lane_info['drift']}")
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Lane detection video saved to {out_vid}")

# Start processing the video
process_video(input_video, output_video)