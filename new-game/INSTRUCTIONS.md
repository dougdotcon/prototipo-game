# SHAPE-SE Game

SHAPE-SE is an interactive camera-based game where players try to fit their body into the shape of Sergipe state.

## Game Overview

- The game starts immediately when launched
- Players have 5 minutes to complete the challenge
- A transparent Sergipe flag is overlaid on the camera feed
- The outline of Sergipe state is displayed with emphasis on the borders
- Players must position themselves to match the shape
- If the player achieves 95% or higher accuracy, they win
- On success, the game displays "Viva Sergipe" and captures the player's image
- On failure, the game displays "Game Over"

## Setup Instructions

### 1. Install Dependencies

```
pip install pygame opencv-python scipy torch torchvision numpy
```

### 2. Download the Body Pose Model

Download the "body_pose_model.pth" file from [this Google Drive link](https://drive.google.com/drive/folders/1Nb6gQIHucZ3YlzVr5ME3FznmF4IqrJzL?usp=sharing) and place it in the `deep_pose` folder.

### 3. Create Game Assets

Run the following scripts to create the necessary game assets:

```
python create_flag_overlay.py  # Create the Sergipe flag overlay
python create_mask.py          # Create the mask for the game
python create_csv.py           # Create the joint positions CSV file
```

### 4. Run the Game

```
python shape_se_runner.py
```

## Controls

- Press ESC to exit the game at any time

## Game Files

- `shape_se_runner.py` - Main entry point for the game
- `shape_se_controller.py` - Game controller for handling camera and user input
- `shape_se_view.py` - Game view for displaying the game interface
- `shape_se_model.py` - Game model for handling game logic and state
- `create_mask.py` - Script to create masks for the game
- `create_csv.py` - Script to create joint position CSV files
- `create_flag_overlay.py` - Script to create the Sergipe flag overlay

## Folder Structure

- `deep_pose/` - Contains the OpenPose implementation
- `images/` - Contains game assets
  - `masks/` - Contains the masks for the game
  - `poses/` - Contains the original poses used to create masks
  - `joints/` - Contains the joint positions CSV files
- `snapshots/` - Contains captured images of successful players
