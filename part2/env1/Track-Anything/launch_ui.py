import os
import sys
import subprocess

def main():
    if "SSL_CERT_FILE" in os.environ:
        del os.environ["SSL_CERT_FILE"]
        
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    track_anything_dir = os.path.abspath(os.path.join(current_dir, "..", "third_party", "Track-Anything"))
    
    if not os.path.exists(track_anything_dir):
        print(f"[Error] Could not find Track-Anything directory at {track_anything_dir}")
        return

    print("Launching Track-Anything UI with safe environment settings...")
    
    try:
        subprocess.run([
            sys.executable, "app.py", 
            "--sam_model_type", "vit_b", 
            "--device", "cuda:0"
        ], cwd=track_anything_dir, check=True)
    except KeyboardInterrupt:
        print("\nUI gracefully closed.")
    except Exception as e:
        print(f"[Error] Failed to launch UI: {e}")

if __name__ == "__main__":
    main()