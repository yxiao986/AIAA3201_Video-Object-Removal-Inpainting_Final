import cv2
import os
import glob
import argparse

def folder_to_video(image_folder, output_video_path, fps=30):
    # Get all PNG or JPG images and sort them by filename (ensuring 00000.png is first)
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")) + 
                    glob.glob(os.path.join(image_folder, "*.jpg")))
    
    if not images:
        print(f"  [Skip] No images found in {image_folder}")
        return False
    
    # Read the first image to obtain the video resolution
    frame = cv2.imread(images[0])
    if frame is None:
        print(f"  [Error] Could not read the first image in {image_folder}")
        return False
        
    height, width, layers = frame.shape
    
    # Initialize the OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4 encoding
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"  -> Fusing {len(images)} images into {os.path.basename(output_video_path)} at {fps} FPS...")
    for image in images:
        video.write(cv2.imread(image))
        
    video.release()
    return True

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert folder(s) of images into MP4 video(s).")
    parser.add_argument("--input_folder", "-i", type=str, required=True, 
                        help="Path to a single sequence folder, OR a parent folder of multiple sequences.")
    parser.add_argument("--fps", type=int, default=24, 
                        help="Frames per second for the output video (default: 24).")
    
    args = parser.parse_args()
    clean_input_folder = os.path.normpath(args.input_folder)
    
    if not os.path.isdir(clean_input_folder):
        print(f"[Error] Directory not found: {clean_input_folder}")
        exit(1)

    # Check if there are images directly in the input folder (indicating single sequence mode)
    images_directly = glob.glob(os.path.join(clean_input_folder, "*.png")) + \
                      glob.glob(os.path.join(clean_input_folder, "*.jpg"))

    # Get all subdirectories (indicating batch mode)
    subdirs = sorted([os.path.join(clean_input_folder, d) for d in os.listdir(clean_input_folder) 
                      if os.path.isdir(os.path.join(clean_input_folder, d))])

    # ==========================================
    # Mode 1: Single Sequence Mode (images directly in the input folder)
    # ==========================================
    if images_directly:
        print(f"Detected Single Sequence in: {clean_input_folder}")
        folder_name = os.path.basename(clean_input_folder)
        parent_dir = os.path.dirname(clean_input_folder)
        output_video = os.path.join(parent_dir, f"{folder_name}.mp4")
        
        if folder_to_video(clean_input_folder, output_video, fps=args.fps):
            print(f"Success! Video saved to: {output_video}")
            
    # ==========================================
    # Mode 2: Batch Mode (e.g., data/DAVIS/JPEGImages/480p)
    # ==========================================
    elif subdirs:
        print(f"Detected Batch Mode. Found {len(subdirs)} subfolders in: {clean_input_folder}")
        success_count = 0
        for subdir in subdirs:
            folder_name = os.path.basename(subdir)

            output_video = os.path.join(clean_input_folder, f"{folder_name}.mp4")
            if folder_to_video(subdir, output_video, fps=args.fps):
                success_count += 1
                
        print(f"\nBatch Processing Complete! Created {success_count}/{len(subdirs)} videos in: {clean_input_folder}")
        
    else:
        print(f"[Error] No images or subdirectories found in {clean_input_folder}!")