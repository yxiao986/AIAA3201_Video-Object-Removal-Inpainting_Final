import cv2
import os
import argparse

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_path = os.path.join(output_dir, f"{frame_idx:05d}.png")
        cv2.imwrite(out_path, frame)
        frame_idx += 1
        
    cap.release()
    print(f"Successfully extracted {frame_idx} frames and saved to: {output_dir}")

if __name__ == "__main__":
    video_file = "data/my_video.mp4" 
    output_folder = "data/my_video"  
    extract_frames(video_file, output_folder)