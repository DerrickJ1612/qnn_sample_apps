# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions in LICENSE.txt are met


import sys
import json
import time

from typing import Dict
from pathlib import Path
from mcp.server import FastMCP

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from utils import PoseModelInference

rubik_root = Path.home()/"Desktop"/"git"/"RubikPI_Warehouse"
model_dir = rubik_root/"models"/"pose_detection_tflite"
delegate_driver = rubik_root/"delegate"/"libQnnTFLiteDelegate.so"
# delegate = tf.lite.experimental.load_delegate(str(delegate_driver),{"backend_type":"htp"})

# Issues with current w8a8 pose detection model, will need to debug or quantize another graph myself
model_name = "hrnet_pose-hrnetpose-float.tflite"

path_2_model = model_dir/model_name

iPose = PoseModelInference(model_path=path_2_model, delegate=(delegate_driver,"htp"))

mcp = FastMCP(
    name = "Rubik MCP Server (FastMCP)",
    host = "0.0.0.0",
    port = 3001)

class RoutineManager:
    """
    Manages yoga/fitness routine execution and pose progression.
    
    The RoutineManager class handles the state and flow of workout routines,
    tracking the current pose, timing, and overall routine status. It provides
    functionality to start, pause, and navigate through pose sequences.
    
    Attributes:
        current_routine: The active routine object containing pose sequences.
            None when no routine is loaded.
        current_pose_index (int): Zero-based index of the current pose within
            the active routine. Defaults to 0.
        routine_active (bool): Flag indicating whether a routine is currently
            running. False when paused or stopped.
        pose_start_time: Timestamp when the current pose began, used for
            timing and duration tracking. None when no pose is active.
    
    Example:
        >>> manager = RoutineManager()
        >>> manager.load_routine(my_routine)
        >>> manager.start_routine()
        >>> manager.next_pose()
    """
    def __init__(self):
        self.current_routine = None
        self.current_pose_index = 0
        self.routine_active = False
        self.pose_start_time = None

routine_manager = RoutineManager()

@mcp.tool()
async def begin_routine(complete_routine: Dict):
    """
    Start a new fitness routine with the first pose.
    
    LLM USAGE: Use when user wants to start a workout. After starting,
    always provide encouraging introduction and explain the first pose.
    
    Args:
        complete_routine (Dict): Routine data with 'routine_name' and 'poses' list.
    
    Returns:
        str: Confirmation message with routine and first pose names.
    """
    routine_manager.current_routine = complete_routine
    routine_manager.current_pose_index = 0
    routine_manager.routine_active = True
    routine_manager.pose_start_time = time.time()
    
    # Start the first pose
    first_pose = complete_routine["poses"][0]
    print(f"Starting pose 1: {first_pose['pose_name']}")
    
    return f"Started {complete_routine['routine_name']} - Now on pose 1: {first_pose['pose_name']}"

@mcp.tool()
async def get_current_pose() -> Dict:
    """Get information about the currently active pose.
    
    IMPORTANT INSTRUCTIONS FOR LLM:
    - The 'detected_pose' field is GROUND TRUTH - always trust what the sensor detects
    - If detected_pose matches target_pose: Give positive encouragement
    - If detected_pose differs from target_pose: Guide user from detected_pose to target_pose
    - Always acknowledge what pose they're currently in based on detected_pose
    - Example: "I see you're in Child's Pose. Let's transition to Cat Cow by..."
    - If no routine started just return the detected pose
    - If prompt is for time left DO NOT speak about current pose, ONLY provide time left until the end
    
    Never question the sensor accuracy - it's always correct."""
    if not routine_manager.routine_active:
        try:
            _, current_pose = iPose.inference(pose_predict=True)
            return {"current_pose": current_pose, "status": "no_routine_active"}
        except Exception as e:
            return f"Error: {str(e)}"
        # return {"status": "no_routine_active"}
    
    _, predicted_pose = iPose.inference(pose_predict=True)
    current_pose = routine_manager.current_routine["poses"][routine_manager.current_pose_index]
    elapsed_time = time.time() - routine_manager.pose_start_time
    remaining_time = current_pose["hold_duration"] - elapsed_time

    expected_pose = current_pose["pose_name"]
    
    current_pose_format = {
        "status": "active",
        "detected_pose": predicted_pose,  # What sensor sees (ground truth)
        "target_pose": expected_pose,     # What routine expects
        "step": current_pose["step"],
        "instructions": current_pose["instructions"],
        "elapsed_time": int(elapsed_time),
        "remaining_time": max(0, int(remaining_time)),
        "total_poses": len(routine_manager.current_routine["poses"]),
        "current_pose_number": routine_manager.current_pose_index + 1,
    }

    return current_pose_format

@mcp.tool()
async def next_pose() -> Dict:
    """
    Move to the next pose in the routine.
    
    LLM USAGE: Use when user says "next", "move on", or indicates readiness 
    to progress. Always announce the new pose name and provide encouragement
    for completing the previous pose.
    
    Returns: Next pose details or routine completion message.
    """
    if not routine_manager.routine_active:
        return {"status": "no_routine_active"}
    
    routine_manager.current_pose_index += 1
    
    if routine_manager.current_pose_index >= len(routine_manager.current_routine["poses"]):
        routine_manager.routine_active = False
        return {"status": "routine_complete", "message": "Congratulations! You've completed your routine!"}
    
    routine_manager.pose_start_time = time.time()
    next_pose = routine_manager.current_routine["poses"][routine_manager.current_pose_index]
    
    next_pose_format = {
        "status": "moved_to_next",
        "target_pose": next_pose["pose_name"],
        "step": next_pose["step"],
        "instructions": next_pose["instructions"],
        "hold_duration": next_pose["hold_duration"]
    }

    return next_pose_format

@mcp.tool()
async def get_pose_keypoints() -> str:
    """
    RAW TECHNICAL DATA: Extracts precise body joint coordinates from computer vision analysis.
    
    ** DO NOT USE FOR POSE IDENTIFICATION ** - This is purely for technical coordinate data.
    
    This tool provides raw numerical keypoint positions (x,y coordinates) for body joints
    like shoulders, elbows, hips, knees, etc. Use ONLY when you need:
    - Technical measurement data
    - Raw coordinate positions 
    - Mathematical analysis of body positioning
    - Custom alignment calculations
    - Engineering/research applications
    
    NOT for: Identifying pose names, yoga guidance, or user-friendly feedback
    
    Returns: JSON array of numerical [x,y] coordinates for each detected body joint
    """
    try:
        keypoints,_ = iPose.inference()
        return json.dumps(keypoints.tolist() if hasattr(keypoints, 'tolist') else keypoints)
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def list_detectable_yoga_poses() -> str:
    """
    Returns list of all yoga poses the system can recognize.
    
    ** AVAILABLE YOGA POSES ** - Get catalog of detectable poses.
    
    Use to discover supported poses before requesting pose identification.
    Includes 9 poses: Downward Dog, Child's, Cobra, Plank, Mountain, 
    Goddess, Warrior I & II, Tree.
    
    Returns: JSON array of pose names in format 'sanskrit_english'
    Example: ['adho_mukha_svanasana_downward_dog', 'balasana_childs_pose', ...]
    """
    try:
        pose_list = iPose.yoga_pose_predictions()
        return json.dumps(pose_list)
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def get_rubik_sw_info() -> dict:
    """
    SYSTEM INFORMATION: Retrieves device and operating system details for Rubik hardware.
    
    ** USE FOR DEVICE/OS QUERIES ** - This is for system and hardware information only.
    
    Provides comprehensive system information about the Rubik Pi device including OS version,
    kernel details, and hardware specifications. Use when you need:
    - Device specifications and capabilities  
    - Operating system version and details
    - System troubleshooting information
    - Hardware compatibility checks
    - Technical support data
    
    NOT for: Pose detection, yoga analysis, or body movement data
    
    Returns: Dictionary with OS info, kernel version, architecture, and system details
    """

    sw_info = {
        "device_type": "Rubik Pi",
        "os_info": "Ubuntu",
        "kernel_version": "223314",
        "architecture": "ARM64",  # Add more relevant info
        "errors": []
    }

    commands = {
        "os_info": "cat /etc/os-release",
        "kernel_version": "uname -a"
    }

    return sw_info

if __name__ == "__main__":
    print("Starting Rubik MCP Server via mcp.run() in SSE mode...")
    mcp.run(transport="streamable-http")

# @mcp.tool()
# async def talk_to_yogi(text_to_say: str) -> str:
#     """
#     AUDIO ONLY: Speaks text aloud using text-to-speech for audio announcements.
#
#     DO NOT use this for creating routines or giving advice. Only use when user explicitly
#     requests audio/speech output like 'say this out loud' or 'announce the pose'.
#
#     Args:
#         text_to_say: The exact text to speak aloud via text-to-speech
#
#     Returns:
#         str: Confirmation message with the announced position
#     """
#     print("Next yoga position")
#
