import numpy as np
import cv2
import os

INPUT_VIDEO = "D:/DIP_Labs/SemesterProject/DIP Project Videos/PXL_20250325_045117252.TS.mp4"
OUTPUT_VIDEO = "D:/DIP_Labs/SemesterProject/output_roi_only.mp4"
OUTPUT_DIR = "D:/DIP_Labs/SemesterProject/output_roi_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

debug_frame_count = 0
MAX_DEBUG_FRAMES = 5

def region_selection(image):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    rows, cols = image.shape[:2]
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
    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Step 4: Apply ROI
    roi_edges = region_selection(edges)

    # Optional: Show or save intermediate outputs for first few frames
    if debug_frame_count < MAX_DEBUG_FRAMES:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{debug_frame_count}_1_gray.jpg"), gray)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{debug_frame_count}_2_blurred.jpg"), blurred)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{debug_frame_count}_3_edges.jpg"), edges)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{debug_frame_count}_4_roi_edges.jpg"), roi_edges)
        debug_frame_count += 1

    # Convert single-channel edges back to 3-channel before writing to color video
    roi_colored = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)

    return roi_colored

def process_video(input_video, output_video):
    if not os.path.exists(input_video):
        print(f"Error: Input video {input_video} not found")
        return
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video resolution: {width}x{height}, FPS: {fps}, Total Frames: {frame_count}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

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
    print(f"ROI-only edge video saved to {output_video}")

if __name__ == "__main__":
    process_video(INPUT_VIDEO, OUTPUT_VIDEO)

