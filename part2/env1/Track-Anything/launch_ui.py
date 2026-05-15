import os
import sys
import subprocess
import argparse


def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_track_anything_dir = os.path.abspath(
        os.path.join(current_dir, "..", "..", "..", "third_party", "Track-Anything")
    )

    parser = argparse.ArgumentParser(description="Launch the Track-Anything Gradio UI.")
    parser.add_argument("--track_anything_dir", default=default_track_anything_dir,
                        help="Path to the local Track-Anything repository.")
    parser.add_argument("--sam_model_type", default="vit_b",
                        help="SAM model type passed to Track-Anything.")
    parser.add_argument("--device", default="cuda:0",
                        help="Device passed to Track-Anything, e.g. cuda:0 or cpu.")
    parser.add_argument("--python_exec", default=sys.executable,
                        help="Python executable used to launch app.py.")
    return parser.parse_args()


def main():
    args = parse_args()

    if "SSL_CERT_FILE" in os.environ:
        del os.environ["SSL_CERT_FILE"]
        
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    track_anything_dir = os.path.abspath(args.track_anything_dir)
    
    if not os.path.exists(track_anything_dir):
        print(f"[Error] Could not find Track-Anything directory at {track_anything_dir}")
        return

    print("Launching Track-Anything UI with safe environment settings...")
    
    try:
        subprocess.run([
            args.python_exec, "app.py",
            "--sam_model_type", args.sam_model_type,
            "--device", args.device,
        ], cwd=track_anything_dir, check=True)
    except KeyboardInterrupt:
        print("\nUI gracefully closed.")
    except Exception as e:
        print(f"[Error] Failed to launch UI: {e}")

if __name__ == "__main__":
    main()
