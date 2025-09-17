import cv2
import os


def extract_frames(video_folder, output_base_folder="extracted_frames"):

    # Create output base folder if it doesn't exist
    os.makedirs(output_base_folder, exist_ok=True)

    # Get all video files in the folder
    video_files = [f for f in os.listdir(video_folder)
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print(f"No video files found in {video_folder}")
        return

    print(f"Found {len(video_files)} videos to process...")

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_folder = os.path.join(output_base_folder, video_name)

        # Create folder for this video's frames
        os.makedirs(output_folder, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open {video_file}, skipping...")
            continue

        print(f"Processing {video_file}...")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame as JPEG
            frame_file = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_file, frame)
            frame_count += 1

        cap.release()
        print(f"Saved {frame_count} frames to {output_folder}")

    print("Frame extraction complete!")

video_folder_path = "D:/DIP_Labs/SemesterProject/DIP Project Videos"
extract_frames(video_folder_path)