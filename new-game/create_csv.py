"""
Create csv files representing joints positions for each mask.
"""
import csv
import os
import cv2
from deep_pose.body import Body

# Create directories if they don't exist
os.makedirs("images/joints", exist_ok=True)

# OpenPose instance used to analyze camera frames.
BODY_ESTIMATION = Body("deep_pose/body_pose_model.pth")

# List of image names to analyze.
MASK_NAMES = ["sergipe_mask"]

def analyze_image(image_name):
    """
    This function analyzes a given image and returns the joint positions
    found within it as a dictionary.

    Args:
        image_name (str): The name of the image to analyze.

    Returns:
        dict: A dictionary of joint positions, where each key is an integer
            representing a joint and each value is a list of doubles
            representing the pixel location of a joint on a given image.
    """
    # Read the image
    image_path = f"images/poses/{image_name}.png"
    image = cv2.imread(image_path)
    
    # If the image doesn't exist, return an empty dictionary
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return {}
    
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Analyze the image for joint positions
    joint_candidates, joint_subsets = BODY_ESTIMATION(image)
    
    # If no joint subsets are found, return an empty dictionary
    if len(joint_subsets) == 0:
        print(f"Error: No joint subsets found in image {image_path}")
        return {}
    
    # Get the joint subset with the highest confidence
    max_subset = joint_subsets[0]
    max_confidence = 0
    for subset in joint_subsets:
        confidence = 0
        for joint in subset:
            if joint != -1:
                confidence += 1
        if confidence > max_confidence:
            max_confidence = confidence
            max_subset = subset
    
    # Parse the joint candidates to find the joint positions
    joint_positions = {}
    for joint_idx, joint in enumerate(max_subset):
        if joint != -1:
            joint_positions[joint_idx] = joint_candidates[int(joint)][0:2]
    
    return joint_positions

def save_joint_positions(image_name, joint_positions):
    """
    This function saves the joint positions to a csv file.

    Args:
        image_name (str): The name of the image.
        joint_positions (dict): A dictionary of joint positions.
    """
    # Create the csv file
    csv_path = f"images/joints/{image_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        for joint_idx, joint_pos in joint_positions.items():
            csv_writer.writerow([joint_idx, joint_pos[0], joint_pos[1]])
    
    print(f"Joint positions saved to {csv_path}")

def main():
    """
    This is the runner function to create and save joint positions.
    """
    for mask_name in MASK_NAMES:
        print(f"Analyzing {mask_name}...")
        joint_positions = analyze_image(mask_name)
        if joint_positions:
            save_joint_positions(mask_name, joint_positions)
        else:
            print(f"Warning: No joint positions found for {mask_name}")

if __name__ == "__main__":
    main()
