import numpy as np
import cv2
import os

# File paths
input_video = "D:/DIP_Labs/SemesterProject/DIP Project Videos/PXL_20250325_043754655.TS.mp4"
output_video = "D:/DIP_Labs/SemesterProject/output_ROI.mp4"
output_dir = "D:/DIP_Labs/SemesterProject/output_ROI_frames"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Counter for saving debug frames
debug_frame_count = 0
max_debug_frames = 5

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
    return masked_image


def frame_processor(frame):
    global debug_frame_count
    roi = region_selection(frame)

    # Save a few debug frames
    if debug_frame_count < max_debug_frames:
        cv2.imwrite(os.path.join(output_dir, f"debug_ROI_frame_{debug_frame_count}.jpg"), roi)
        debug_frame_count += 1

    return roi

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

        processed = frame_processor(frame)
        out.write(processed)

        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx}/{frame_count}")
        frame_idx += 1

    cap.release()
    out.release()
    print(f"ROI-only video saved to {out_vid}")

# Start processing the video
process_video(input_video, output_video)
