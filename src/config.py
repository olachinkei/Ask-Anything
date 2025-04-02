import os
from pathlib import Path

# Project settings
PROJECT_NAME = "wandb-japan/mvbench-weave-evaluation-volume-test3"

# Base paths - these should be configured by the user
BASE_DATA_DIR = Path("../data/MVBench")  # Update this path
JSON_DIR = BASE_DATA_DIR / "json"
VIDEO_DIR = BASE_DATA_DIR / "video"

# Dataset configurations
DATA_LIST = {
    "Action Sequence": ("action_sequence.json", "star/star/Charades_v1_480/", "video", True),
    "Action Prediction": ("action_prediction.json", "star/star/Charades_v1_480/", "video", True),
    "Action Antonym": ("action_antonym.json", "ssv2_video/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "star/star/Charades_v1_480/", "video", True),
    "Object Shuffle": ("object_shuffle.json", "perception/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "sta/sta/sta_video/", "video", True),
    "Scene Transition": ("scene_transition.json", "scene_qa/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "perception/perception/videos/", "video", False),
    "Character Order": ("character_order.json", "perception/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/vlnqa", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/tvqa/frames_fps3_hq/", "frame", True),
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/clevrer/video_validation/", "video", False),
}

# Model settings
NUM_SEGMENTS = 8
RESOLUTION = 224
INPUT_MEAN = [0.48145466, 0.4578275, 0.40821073]
INPUT_STD = [0.26862954, 0.26130258, 0.27577711]

# System prompts
SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. Based on "
    "your observations, select the best option that accurately addresses the question.\n"
)

ANSWER_INSTRUCTION_PROMPT = "\nOnly give the best option.\n" 