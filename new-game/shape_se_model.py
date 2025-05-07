"""
This is the model for the SHAPE-SE game. This script relies on a
public github repository (pytorch-openpose) in order to analyze inputted camera
frames for human body positions. Open Pose is a machine learning algorithm
that can analyze an image for joint positions in terms of pixel locations for
each joint. This repository can be found at this link:
https://github.com/Hzzone/pytorch-openpose
"""
import csv
import random
from cv2 import cv2 as cv
import numpy as np
from deep_pose.body import Body


class ShapeSEGame:
    """
    SHAPE-SE game model with helper functions that dictate gameflow.

    Attributes:
        BODY_ESTIMATION (Body): Body estimation object from open pose.
        MASK_NAMES (list): List of names of each mask, represented as strings.
        _mask_and_joints (list): List of tuples, where each tuple contains the
            string file path to the mask that a user should fit into and a
            string file path to the csv that stores the joint positions users
            need to match.
        _joint_positions (dict): Dictionary of joint positions, where each key
            is an integer representing a joint (joint to integer conversions
            can be found in the openpose github) and each value is a list of
            doubles representing the pixel location of a joint on a given
            image.
        _joint_candidates (list): 2-D list of all joints, their positions, and
            the confidence of the open pose neural network, for every joint
            found in a given image.
        _joint_subsets (list): 2-D list of all joint subsets, where each subset
            is a list of joints that are connected to each other.
        _current_score (int): The current score of the user.
        _total_score (int): The total score of the user.
    """

    # Instance of Body class from open pose that will be used to analyze frames.
    BODY_ESTIMATION = Body("deep_pose/body_pose_model.pth")

    # List of each mask that will be available for users to play with.
    MASK_NAMES = ["sergipe_mask"]

    def __init__(self):
        """
        This is the constructor for the ShapeSEGame class. The constructor
        creates _mask_and_joints based on the class attribute of all the mask
        names, initializes the variables that map joint positions to empty
        lists/dictionaries and initializes the score variables to 0.
        """
        # Assembles tuples that store paths to the image mask and the joint
        # positions csv and stores them to a list.
        self._mask_and_joints = []
        for mask in self.MASK_NAMES:
            # Each image can be found in the images/masks folder.
            frame = cv.imread(f"images/masks/{mask}.png")
            # By default. OpenCV stores images as BGR and need to be converted
            # to properly display them in pygame.
            cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # Images need to be resized to ensure images fit the screen.
            frame = cv.resize(frame, (640, 480))
            # Each joint position can be found in the images/joints folder.
            joint_path = f"images/joints/{mask}.csv"
            self._mask_and_joints.append((frame, joint_path))

        # Initialize joint positions to an empty dictionary.
        self._joint_positions = {}
        # Initialize joint candidates and subsets to empty lists.
        self._joint_candidates = []
        self._joint_subsets = []
        # Initialize current and total score to 0.
        self._current_score = 0
        self._total_score = 0

    @property
    def joint_candidates(self):
        """
        Return the joint candidates.
        """
        return self._joint_candidates

    @property
    def joint_subsets(self):
        """
        Return the joint subsets.
        """
        return self._joint_subsets

    @property
    def current_score(self):
        """
        Return the current score.
        """
        return self._current_score

    @property
    def total_score(self):
        """
        Return the total score.
        """
        return self._total_score

    def get_random_mask(self):
        """
        This function returns a random mask from the list of masks.

        Returns:
            tuple: A tuple containing the mask and the joint positions csv.
        """
        return random.choice(self._mask_and_joints)

    def analyze_frame(self, frame):
        """
        This function analyzes the inputted frame for potential joints and
        stores them to _joint_candidates and _joint_subsets by calling the
        open pose object within this class.

        Args:
            frame (numpy.ndarray): A 3-D numpy array that represents the RGB
                values of the frame to be analyzed by open pose. This frame
                array should be of size 480x640x3.
        """
        self._joint_candidates, self._joint_subsets =\
            self.BODY_ESTIMATION(frame)

    def parse_for_joint_positions(self):
        """
        This function parses the joint candidates and subsets to find the
        joint positions of the user. This function should be called after
        analyze_frame.
        """
        # If no joint subsets are found, return.
        if len(self._joint_subsets) == 0:
            return

        # Get the joint subset with the highest confidence.
        max_subset = self._joint_subsets[0]
        max_confidence = 0
        for subset in self._joint_subsets:
            confidence = 0
            for joint in subset:
                if joint != -1:
                    confidence += 1
            if confidence > max_confidence:
                max_confidence = confidence
                max_subset = subset

        # Parse the joint candidates to find the joint positions.
        for joint_idx, joint in enumerate(max_subset):
            if joint != -1:
                self._joint_positions[joint_idx] = \
                    self._joint_candidates[int(joint)][0:2]

    def compute_accuracy(self, joint_path):
        """
        This function computes the accuracy of the user's pose compared to the
        mask. This function should be called after parse_for_joint_positions.

        Args:
            joint_path (str): The path to the csv file containing the joint
                positions of the mask.

        Returns:
            float: The accuracy of the user's pose compared to the mask.
        """
        # If no joint positions are found, return 0.
        if len(self._joint_positions) == 0:
            return 0

        # Read the joint positions from the csv file.
        mask_joint_positions = {}
        try:
            with open(joint_path, "r", encoding="utf-8") as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    mask_joint_positions[int(row[0])] = \
                        [float(row[1]), float(row[2])]
        except FileNotFoundError:
            # If the csv file is not found, return 0.
            return 0

        # Compute the accuracy of the user's pose compared to the mask.
        total_distance = 0
        num_joints = 0
        for joint_idx, joint_pos in self._joint_positions.items():
            if joint_idx in mask_joint_positions:
                mask_joint_pos = mask_joint_positions[joint_idx]
                distance = np.sqrt((joint_pos[0] - mask_joint_pos[0]) ** 2 +
                                  (joint_pos[1] - mask_joint_pos[1]) ** 2)
                total_distance += distance
                num_joints += 1

        # If no joints are found, return 0.
        if num_joints == 0:
            return 0

        # Compute the average distance.
        avg_distance = total_distance / num_joints
        # Compute the accuracy based on the average distance.
        # The accuracy is 100 if the average distance is 0, and 0 if the
        # average distance is 100 or more.
        accuracy = max(0, 100 - avg_distance)
        # Update the current score.
        self._current_score = accuracy
        # Update the total score.
        self._total_score = accuracy
        return accuracy

    def is_success(self):
        """
        This function returns whether the user has succeeded in the game.
        Success is defined as having a total score of 95 or higher.

        Returns:
            bool: Whether the user has succeeded in the game.
        """
        return self._total_score >= 95
