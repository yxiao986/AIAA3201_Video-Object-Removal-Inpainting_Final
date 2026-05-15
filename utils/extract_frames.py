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


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ordered frames from a video file.")
    parser.add_argument("--video_path", "-i", required=True, help="Input video path.")
    parser.add_argument("--output_dir", "-o", required=True, help="Directory for extracted frames.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames(args.video_path, args.output_dir)
