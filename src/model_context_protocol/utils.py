# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions in LICENSE.txt are met

"""
TODO: 
    1. Add Logging
    2. Unit Tests
    3. For single predictions, apply temporal smoothing across 5 frames and select the mode prediction
"""

import warnings
import os
import json
import cv2 as cv
import tensorflow as tf
import numpy as np

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

from pathlib import Path
from typing import Tuple, List, Optional, Dict
from joblib import load
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TextStyle:
    org: tuple[int, int]
    fontFace: int = cv.FONT_HERSHEY_DUPLEX
    fontScale: float = 0.4
    color: tuple[int, int, int] = (0,0,255)
    thickness: int = 1

class PoseModelInference():

    def __init__(self, model_path: Path, delegate: Optional[Tuple[str,str]]=None):
        """
        Initialize pose detection system with TensorFlow Lite model and pose classifier.
        
        Sets up dual-model system: TFLite for keypoint detection and joblib classifier
        for pose identification. Loads models, decoder, and configures hardware acceleration.
        
        Args:
            model_path (Path): Path to TensorFlow Lite model file (.tflite)
            delegate (Optional[Tuple[str,str]]): Hardware acceleration as (library_path, backend_type).
                Example: ("/usr/lib/libedgetpu.so.1", "EDGETPU"). Falls back to CPU if fails.
        
        Raises:
            ValueError: If model_path file doesn't exist
            
        Side Effects:
            - Loads pose classifier from ~/Desktop/git/yogi/models/
            - Initializes TFLite interpreter with optional hardware delegate
            - Sets up input/output tensor configurations
        """
        if not model_path.is_file():
            raise ValueError(f"Model file not found: {model_path}")
        
        root_dir = Path.home()
        model_path = str(model_path) 
        keypoint_model_dir = root_dir/"Desktop"/"git"/"yogi"/"models"
        keypoint_model = "yogi_inference_model_v0107_acc-75.joblib"
        keypoint_decoder = "yogi_decoder.json"
        keypoint_model_path = keypoint_model_dir/keypoint_model
        keypoint_decoder_path = keypoint_model_dir/keypoint_decoder

        self.keypoint_inference = load(keypoint_model_path)
        with open(str(keypoint_decoder_path),"r") as file:
            self.inference_decoder = json.load(file)
        self.inference_decoder = {idx: val[1].split("\\")[-1] for idx,val in enumerate(self.inference_decoder.items())}


        if delegate:
            try:
                loaded_delegate = tf.lite.experimental.load_delegate(
                    delegate[0],
                    options={"backend_type": delegate[1]})
                
                self.interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[loaded_delegate]
                )
            except Exception as e:
                print(f"Failed to load delegate: {e}")
                delegate = None

        if not delegate:
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path
            )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0].get("shape")
        self.output_shape = self.output_details[0].get("shape")

    def model_details(self):
        """
        Print comprehensive model configuration and capabilities.
        
        Displays TensorFlow Lite model specs (input/output shapes, types), 
        pose classifier parameters, and list of detectable yoga poses.
        
        Outputs:
            - TFLite input/output tensor specifications
            - Keypoint inference model parameters  
            - Available yoga pose names (formatted for readability)
        
        Use for debugging, model verification, or discovering system capabilities.
        """
        for layer in self.input_details:
            print(f"Expected Input Name: {layer.get('name')}")
            print(f"Expected Input Shape: {layer.get('shape')}")
            print(f"Expected Input Type: {layer.get('dtype')}")

        print("*"*50)

        for layer in self.output_details:
            print(f"Expected Output Name: {layer.get('name')}")
            print(f"Expected Output Shape: {layer.get('shape')}")
            print(f"Expected Output Type: {layer.get('dtype')}")

        print("*"*50)

        print("Keypoint Inference Model\n")
        for key, val in self.keypoint_inference.get_params().items():
            print(f"{key}: {val}")

        print("")
        print("Available Yoga Poses:\n")
        for key, val in self.inference_decoder.items():
            print(f"{val.replace('_',' ')}")

    
    def inference(self, camera: int=0, display: bool=False, save_frames: bool=False, pose_predict: bool=False, img_max: Optional[int]=5) -> Optional[List]:                                                             
        """
        Perform pose detection inference using camera input with optional pose classification.
        
        Captures frames from camera, processes through TensorFlow Lite model for keypoint detection,
        and optionally classifies yoga poses. Supports real-time display, frame saving, and 
        batch processing modes.
        
        Args:
            camera (int): Camera index for OpenCV capture (default: 0)
            display (bool): If True, shows real-time video with keypoints until 'q' pressed.
                        If False, processes img_max frames and returns last result.
            save_frames (bool): Save processed frames with keypoints to disk
            pose_predict (bool): Enable yoga pose classification and display prediction text
            img_max (int): Number of frames to process in non-display mode (default: 5)
        
        Returns:
            tuple: (keypoints, pose_prediction) when display=False
                (None, None) when display=True
            keypoints: List of (y,x) coordinate tuples for detected body joints
            pose_prediction: String name of classified yoga pose (if pose_predict=True)
        
        Raises:
            ValueError: If camera cannot be opened or frame capture fails
        
        Notes:
            - Automatically handles model input/output scaling and coordinate transformations
            - In display mode, press 'q' to quit
            - Saved frames stored in RubikPI_Warehouse/other/ directory
        """
        cap = cv.VideoCapture(camera)
        if not cap.isOpened():                                                                                 
            raise ValueError(f"Error while trying to open camera - {self.available_cameras}")   
        
        input_image_height, input_image_width = self.input_shape[1], self.input_shape[2]
        heatmap_height, heatmap_width = self.output_shape[1], self.output_shape[2]
        scaler_height = input_image_height/heatmap_height
        scaler_width = input_image_width/heatmap_width

        pose = None
        counter = 0
        frame_max = int(1e5) if display else img_max

        try:
            while counter < frame_max:
                ret, frame = cap.read() # frame => (H,W,C) => (480, 640, 3)
                if not ret:
                    raise ValueError("Can't receive frame. Exiting....")
                
                processed_frame, dtype = self._transform_numpy_opencv(frame, self.input_shape)
                inference_frame = np.expand_dims(processed_frame, axis=0) #inference_frame => (B,H,W,C) => (1, 256, 192,3)

                ###################### TFLITE Inference ###################################
                self.interpreter.set_tensor(self.input_details[0]["index"], inference_frame)
                self.interpreter.invoke()
                output_tensor = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
                ###########################################################################

                if dtype == np.uint8:
                    output_tensor = self._dequantize(output_tensor)

                frame_height, frame_width = frame.shape[0], frame.shape[1]
                frame_scaler_height = frame_height/input_image_height
                frame_scaler_width = frame_width/input_image_width
                keypoint_coordinate_list, yoga_prediction = self.FIX_keypoint_processor_numpy(post_inference_array=output_tensor,
                                                                             heatmap_to_input_scaler_height=scaler_height,
                                                                             heatmap_to_input_scaler_width=scaler_width,
                                                                             input_to_original_scaler_height=frame_scaler_height,
                                                                             input_to_original_scaler_width=frame_scaler_width,
                                                                             predict=pose_predict
                                                                        )
                if pose_predict:
                    style = TextStyle(org=(10,30))
                    cv.putText(frame, yoga_prediction, **style.__dict__)
                    print(yoga_prediction)
                
                if display:
                    for (y,x) in keypoint_coordinate_list:
                        cv.circle(frame, (x,y), radius=3, color=(0,0,255), thickness=-1)

                    cv.imshow("Yogi", frame)
                    if cv.waitKey(1) == ord("q"):
                        break
                    
                if save_frames:
                    save_dir = Path(__file__).parent.parent/"RubikPI_Warehouse"/"other"
                    time_stamp = datetime.now().strftime("%m%d%Y%M%S")
                    for (y,x) in keypoint_coordinate_list:
                        cv.circle(frame, (x,y), radius=3, color=(0,0,255), thickness=-1)

                    # Save instead of show
                    cv.imwrite(save_dir/f"frame_{counter}_{time_stamp}.jpg", frame)
                    print(f"Saved frame_{counter}_{time_stamp}.jpg")
                
                counter += 1

        finally:
            cap.release()
            if display:
                cv.destroyAllWindows()

        return (keypoint_coordinate_list, yoga_prediction) if not display else (None, None)
    
    def PLACEHOLDER_get_stabalized_prediction(self):
        """
        Apply temporal smoothing with majority voting across 5-frame window.
        
        Reduces prediction noise by maintaining rolling buffer of recent
        predictions and returning the most frequently predicted class.
        """

    def yoga_pose_inference(self, keypoints: np.array) -> str:
        """
        Classify yoga pose from keypoint coordinates.
        
        Args:
            keypoints (np.array): Body joint coordinates from pose detection
        
        Returns:
            str: Decoded pose name (e.g., "downward_dog", "childs_pose")
        """
        prediction = self.keypoint_inference.predict(np.array(keypoints).reshape(1,-1))
        prediction_decode = self.inference_decoder[prediction[0]]
        return prediction_decode
    
    def yoga_pose_predictions(self) -> Dict[int,str]:
        """
        Get mapping of all detectable yoga poses.
        
        Returns:
            Dict[int,str]: Pose index to name mapping from decoder
        """
        return self.inference_decoder

    
    def _dequantize(self, quantized_outputs: np.array):
        """
        Convert quantized TensorFlow Lite outputs to float32.
        
        Applies model-specific scale and zero-point parameters to restore
        original value range from quantized int8 outputs.
        
        Args:
            quantized_outputs (np.array): Raw quantized model outputs
        
        Returns:
            np.array: Dequantized float32 tensor
        """
        scale = self.output_details[0].get("quantization_parameters").get("scales")[0]
        zero_point = self.output_details[0].get("quantization_parameters").get("zero_points")[0]

        dequantized = scale * (quantized_outputs.astype(np.float32) - zero_point)
        return dequantized
    
    def _transform_numpy_opencv(self, image: np.ndarray,
                                expected_shape
                                ) -> Tuple[np.ndarray, np.ndarray, np.dtype]:
        """
        Preprocess image for TensorFlow Lite model inference.
        
        Resizes image to model input dimensions and normalizes to [0,1] range.
        Handles both quantized (uint8) and float32 model input types automatically.
        
        Args:
            image (np.ndarray): Input image in HWC format (uint8)
            expected_shape: Model input shape tuple (N, H, W, C) for dimension extraction
        
        Returns:
            tuple: (original_frame, processed_frame, dtype)
                - original_frame: Resized image in HWC format
                - processed_frame: Same as original_frame (ready for model input)
                - dtype: Model's expected input type (uint8 or float32)
        
        Notes:
            - Uses INTER_CUBIC interpolation for resizing
            - Automatically detects model input type from TFLite metadata
            - Normalization applied regardless of target dtype
        """
        d_type = np.uint8 if self.input_details[0].get("dtype")==np.uint8 else np.float32

        height, width = expected_shape[1], expected_shape[2]
        resized_image = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)
        processed_image = resized_image.astype(d_type) / 255

        return (processed_image, d_type)


    def _keypoint_processor_numpy(self, post_inference_array: np.ndarray,
                                    scaler_height: int,
                                    scaler_width: int
                                    ) -> List[Tuple[int, int, type]]:
        """
        Extracts keypoint coordinates from heatmaps and scales them to match the original image dimensions.

        Parameters:
        -----------
        post_inference_array : np.ndarray
            A 3D array of shape (num_keypoints, heatmap_height, heatmap_width),
            containing the model's predicted heatmaps for each keypoint.

        scaler_height : int
            Scaling factor for the height dimension to map from heatmap space to original image space.

        scaler_width : int
            Scaling factor for the width dimension to map from heatmap space to original image space.

        Returns:
        --------
        list of tuple
            A list of (y, x) coordinates (as integers) representing the scaled keypoint positions
            in the original image space.
        """
        keypoint_coordinates = []

        for keypoint in range(post_inference_array.shape[2]):
            heatmap = post_inference_array[:, :, keypoint]
            max_val_index = np.argmax(heatmap)
            img_height, img_width = np.unravel_index(max_val_index, heatmap.shape)
            coords = (int(img_height * scaler_height), int(img_width * scaler_width))
            keypoint_coordinates.append(coords)

        return keypoint_coordinates
    
    def FIX_keypoint_processor_numpy(self, post_inference_array: np.ndarray, 
                             heatmap_to_input_scaler_height: int, heatmap_to_input_scaler_width: int,
                             input_to_original_scaler_height: int, input_to_original_scaler_width: int,
                            predict: bool=True) -> List[Tuple[int, int]]:
        """
        Extract keypoint coordinates from heatmaps and optionally classify yoga pose.
        
        Processes model output heatmaps to find peak activations for each body joint,
        then scales coordinates through the complete transformation chain: 
        heatmap → model input → original image dimensions.
        
        Args:
            post_inference_array (np.ndarray): Model output heatmaps, shape (H, W, num_keypoints)
            heatmap_to_input_scaler_height (int): Height scaling from heatmap to model input
            heatmap_to_input_scaler_width (int): Width scaling from heatmap to model input  
            input_to_original_scaler_height (int): Height scaling from model input to original image
            input_to_original_scaler_width (int): Width scaling from model input to original image
            predict (bool): If True, perform yoga pose classification on extracted keypoints
        
        Returns:
            tuple: (keypoint_coordinates, pose_prediction)
                - keypoint_coordinates: List of (y, x) tuples in original image space
                - pose_prediction: Yoga pose name string if predict=True, None otherwise
        
        Notes:
            - Uses argmax to find peak activation in each keypoint heatmap
            - Applies cascaded coordinate transformations for accurate positioning
            - Flattens keypoint coordinates for pose classification input
        """
        keypoint_coordinates = []
        predict_coords = []
        prediction_decode = None

        for keypoint in range(post_inference_array.shape[2]):
            heatmap = post_inference_array[:,:,keypoint]
            max_val_index = np.argmax(heatmap)
            img_height, img_width = np.unravel_index(max_val_index, heatmap.shape)
            
            if predict:
                predict_coords.append(int(img_height))
                predict_coords.append(int(img_width))
                                    
            coords = (round(img_height * heatmap_to_input_scaler_height*input_to_original_scaler_height), 
                        round(img_width * heatmap_to_input_scaler_width*input_to_original_scaler_width))
            keypoint_coordinates.append(coords)
            
        if predict: 
            prediction_decode = self.yoga_pose_inference(keypoints=np.array(predict_coords))
        return keypoint_coordinates, prediction_decode
    
    @property
    def available_cameras(self, max_cameras: int=5) -> List[int]:
        """
        Lists available camera indices.

        Parameters:
        ----------
        max_cameras : int, optional
            Maximum number of cameras to check (default is 10).

        Returns:
        -------
        str
            A string listing available camera indices.
        """        
        #Log suppression is currently not working, need to figure out why
        os.environ["OPENCV_LOG_LEVEL"] = "FATAL"                                                # Suppress logging to ignore any error that's not fatal

        available_cameras: List[int] = []
        for cam in range(max_cameras):
            cap = cv.VideoCapture(cam)
            if cap.isOpened():
                available_cameras.append(str(cam))
                cap.release()

        os.environ["OPENCV_LOG_LEVEL"] = "INFO"                                                 # Restore logging back to normal

        return f"Available Cameras: {' | '.join(available_cameras)}"
    
if __name__=="__main__":
    base_path = Path.home()
    model_dir = base_path/"Desktop"/"git"/"RubikPI_Warehouse"/"models"/"pose_detection_tflite"
    # model_name = "hrnet_pose-hrnetpose-w8a8.tflite" #"hrnet_pose-hrnetpose-float.tflite"
    
    model_name = "hrnet_pose-hrnetpose-float.tflite"
    model_path = model_dir/model_name
    delegate_dir = base_path/"Desktop"/"git"/"RubikPI_Warehouse"/"delegate"
    delegate_path = delegate_dir/"libQnnTFLiteDelegate.so"
    processor = "htp"
    delegate_options = None #(str(delegate_path),processor)
    iInfer = PoseModelInference(model_path=model_path,delegate=delegate_options)
    iInfer.model_details()
    print(iInfer.inference(display=True, save_frames=False, pose_predict=True, img_max=5))

