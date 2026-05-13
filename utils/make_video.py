import cv2
import os
import glob
import argparse

def folder_to_video(image_folder, output_video_path, fps=30):
    # Get all PNG or JPG images and sort them by filename (ensuring 00000.png is first)
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")) + 
                    glob.glob(os.path.join(image_folder, "*.jpg")))
    
    if not images:
        print(f"[Error] No images found in {image_folder}!")
        return
    
    # Read the first image to obtain the video resolution
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    # Initialize the OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4 encoding
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Fusing {len(images)} images into a video at {fps} FPS...")
    for image in images:
        video.write(cv2.imread(image))
        
    video.release()
    print(f"✅ Success! Video saved to: {output_video_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert a folder of images into an MP4 video.")
    parser.add_argument("--input_folder", "-i", type=str, required=True, 
                        help="Path to the folder containing image frames.")
    parser.add_argument("--fps", type=int, default=24, 
                        help="Frames per second for the output video (default: 24).")
    
    args = parser.parse_args()
    
    # Clean up the path (removes trailing slashes if the user accidentally adds them, e.g., 'data/bmx-trees/')
    clean_input_folder = os.path.normpath(args.input_folder)
    
    # Extract the base folder name (e.g., "bmx-trees" from "data/bmx-trees")
    folder_name = os.path.basename(clean_input_folder)
    
    # Extract the parent directory (e.g., "data" from "data/bmx-trees")
    parent_dir = os.path.dirname(clean_input_folder)
    
    # Construct the output video path in the same location with the same name
    output_video = os.path.join(parent_dir, f"{folder_name}.mp4")
    
    # Execute the conversion
    folder_to_video(clean_input_folder, output_video, fps=args.fps)