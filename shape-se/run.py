import json
import os
import sys
from core.vision import VisionProcessor
from core.game import Game

def load_config():
    """Load configuration from config.json."""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Config file not found. Using default settings.")
        return {
            "camera_id": 0,
            "threshold": 95,
            "shape_path": "assets/shape-se.png",
            "background_path": "assets/flag-se.jpg",
            "mirror_mode": True
        }

def main():
    """Main function to run the game."""
    # Change working directory to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory set to: {script_dir}")

    # Load configuration
    config = load_config()
    print(f"Loaded config: {config}")

    # Check if contorno file exists
    contorno_path = os.path.join(script_dir, "contorno-mapa-SE.png")
    print(f"Checking for contorno file at: {contorno_path}")
    if os.path.exists(contorno_path):
        print(f"Contorno file exists: {contorno_path}")
    else:
        print(f"Contorno file NOT found: {contorno_path}")

    try:
        # Initialize vision processor
        vision = VisionProcessor(
            camera_id=config.get("camera_id", 0),
            mirror_mode=config.get("mirror_mode", True)
        )

        # Initialize and run game
        game = Game(vision, config)
        game.run()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
