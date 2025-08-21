import onnxruntime as ort
import os
import json
import platform

from pathlib import Path
from typing import Dict, List

import sys
# print(ort.get_all_providers())

class ModelLoader:
    def __init__(self, model: str, processor: str, model_type: str) -> None:
        """
        Initializes an instance with the specified model name, processor type, and model type.

        This constructor sets up paths and configurations required for model execution.
        It determines the platform, loads model and execution configurations, and resolves
        paths to relevant directories and resources such as ONNX Runtime.

        Args:
            model (str): The name of the model to be used.
            processor (str): The type of processor to target (e.g., "cpu", "npu").
            model_type (str): 
                Indicates the variant of the model:
                - For most models: "Default" or "Quantized".
                - For LLMs: specifies model size (quantized by default).

        Attributes:
            processor (str): Uppercased processor type for normalization.
            model (str): The name of the specified model.
            model_type (str): Type or variant of the model.
            root_dir (Path): The current working directory of the environment.
            onnx_root (Path): The resolved path to the ONNX Runtime installation.
            model_config (dict): Parsed configuration data from models.json.
            executioner_config (dict): Parsed configuration data from executioner.json.
            model_subdirectory_path (Path): Path to the model’s subdirectory, as resolved 
                by get_model_path_subdirectory().
        """
        self.processor = processor.upper()
        self.model = model
        self.model_type = model_type
        self.root_dir = Path.cwd()
        self.onnx_root = Path(ort.__file__).parent.resolve()

        self.platform_manager()

        self.model_config = self._get_config(filename="models.json")
        self.executioner_config = self._get_config(filename="executioner.json")     

        self.model_subdirectory_path = self.get_model_path_subdirectory(model_name=self.model)
        
        

    def _get_config(self, filename: str) -> json:
        """
        Constructs the absolute path to a configuration file based on its filename.

        Args:
            filename (str): The name of the configuration file.

        Returns:
            str: The absolute path to the configuration file.
        """
        config_path = self.root_dir/filename

        with open(config_path, "r") as f:
            config = json.load(f)

        return config

    def get_model_path_subdirectory(self, model_name: str) -> ort.InferenceSession:
        """
        Retrieves the absolute path to a model subdirectory based on the provided model name.

        This method looks up the model name in the `models.json` configuration file and returns
        the resolved path to its subdirectory under the local "models/" directory.

        Args:
            model_name (str): The name of the model to load. This should correspond to a key 
                            in the "MODELS" section of the models.json configuration file.

        Returns:
            Path: The absolute path to the model's subdirectory.

        Raises:
            ValueError: If the model name is not found in the configuration or the subdirectory path is missing.
        """

        
        model_subdirectory = self.model_config["MODELS"][model_name.upper()].get("PATH_SUBDIRECTORY")
        model_subdirectory_path = self.root_dir/"models"/model_subdirectory 

        if not model_subdirectory_path:
            raise ValueError(f"Model path for {model_name} not found in config")

        return model_subdirectory_path

    def _get_dll_path(self, runtime: str="qnn") -> str:
        """
        Retrieves the absolute path to the DLL file for the current processor.

        This method looks up the DLL path from the executioner.json configuration file 
        using the current processor type (e.g., "CPU", "NPU"). The path is resolved 
        relative to the ONNX Runtime installation directory.

        Returns:
            str: The absolute path to the DLL file as a string.

        Raises:
            KeyError: If the processor type is not found in the executioner.json configuration file.
        """

        dll_path = self.onnx_root/self.executioner_config[self.processor].get("PATH")
        
        return dll_path
    
    def _get_executioner(self) -> Dict[str,str]:
        """
        Retrieves the ExecutionProvider configuration for a specified processor.

        This function loads a configuration file (executioner.json) containing available processors 
        and their corresponding ExecutionProvider. It checks if the specified processor is available 
        in the configuration file. If the processor is not listed, it raises an error indicating 
        the available options. If the processor is found, it returns the relevant ExecutionProvider 
        for the selected processor.

        Returns:
            Dict[str, str]: A dictionary containing execution provider settings for the processor.

        Raises:
            ValueError: If the processor is not found in the executioner.json configuration file.
        """
       
        if self.processor not in self.executioner_config.keys():
            raise ValueError(f"Selected processor ({self.processor}) not available. Please select {' | '.join(list(self.executioner_config.keys()))}")
        return self.executioner_config[self.processor]


    def load_model(self, onnx_graph: ort, htp_performance_mode: str="burst", 
                   soc_model: str="60", profiling_level: str="off",
                   profiling_file_path: str=None, 
                   htp_graph_finalization_optimization_mode: str="3") -> ort.InferenceSession:
        """
        Loads an ONNX model and configures an InferenceSession with QNN execution provider options.

        This method constructs the full model path from the provided graph name, retrieves the execution 
        provider settings and DLL path, and sets up ONNX Runtime session options. It configures 
        QNN-specific provider options such as performance mode, SoC model, profiling, and graph 
        finalization behavior, then returns a ready-to-use InferenceSession.

        Args:
            onnx_graph (ort): The filename of the ONNX model (e.g., "model.onnx").
            htp_performance_mode (str): HTP performance mode (e.g., "burst", "balanced","sustained_high_performance").
            soc_model (str): Target SoC model identifier (e.g., "60" for Snapdragon SoC 60).
            profiling_level (str): Profiling verbosity level (e.g., "off", "basic", "detailed").
            profiling_file_path (str, optional): Path to write profiling results. Defaults to model directory.
            htp_graph_finalization_optimization_mode (str): Graph optimization level (e.g., "1", "2", "3").

        Returns:
            ort.InferenceSession: An ONNX Runtime InferenceSession configured with the specified QNN provider.

        Raises:
            ValueError: If processor configuration is missing or invalid.
        """
        session_options = ort.SessionOptions()

        # CPU Optimizations
         # CPU Optimization Options
        session_options.intra_op_num_threads = 0  # 0 = use all available cores, or set specific number
        session_options.inter_op_num_threads = 0  # 0 = use all available cores
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # or ORT_PARALLEL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Memory optimizations
        session_options.enable_mem_pattern = True
        session_options.enable_mem_reuse = True
        session_options.enable_cpu_mem_arena = True
        
        # Profiling and logging
        session_options.enable_profiling = False  # Set to True for performance analysis
        session_options.log_severity_level = 3  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

         # Additional optimizations
        session_options.add_session_config_entry("session.disable_prepacking", "0")  # Enable weight prepacking
        session_options.add_session_config_entry("optimization.enable_gelu_approximation", "1")
        
        model_path = self.model_subdirectory_path/onnx_graph
        executioner = self._get_executioner()
        dll_path = self._get_dll_path()

        qnn_provider_options = {
            "backend_path": dll_path, 
            "htp_performance": htp_performance_mode,
            "soc_model": soc_model,
            "profiling_level": profiling_level,
            "profiling_file_path": str(self.model_subdirectory_path) if profiling_file_path == None else str(profiling_file_path),
            "htp_graph_finalization_optimization_mode": htp_graph_finalization_optimization_mode,
            "enable_htp_shared_memory_allocator": 1,
            "qnn_context_priority": "high",
            "offload_graph_io_quantization": 1
        }

        session = ort.InferenceSession(model_path, 
                                       providers=[(executioner.get("EP"),qnn_provider_options)],
                                       sess_options=session_options
                                       )

        return session
    
    @property
    def graphs(self) -> str:
        """
        Retrieves the ONNX graph filename for the current model and model type.

        This method looks up the appropriate ONNX graph name from the models.json 
        configuration file using the current model name and model type.

        Returns:
            str: The filename of the ONNX graph (e.g., "model_qnn.onnx").

        Raises:
            KeyError: If the model or model type is not found in the configuration.
        """

        return self.model_config["MODELS"][self.model.upper()][self.model_type.upper()]
    
    def platform_manager(self):
        """
        Determines the system architecture and operating system, and adjusts processor settings accordingly.

        This method sets `self.arch` and `self.system` based on the current platform. If the environment 
        is not ARM64 or AArch64 (i.e., not suitable for Hexagon DSP inference), it forces the processor 
        type to "CPU" to ensure compatibility.

        Modifies:
            self.arch (str): System architecture in lowercase (e.g., "x86_64", "arm64").
            self.system (str): Operating system in uppercase (e.g., "WINDOWS", "LINUX").
            self.processor (str): Overridden to "CPU" if incompatible with Hexagon inference.
        """
        self.arch = platform.machine().lower()
        self.system = platform.system().upper()

        # Inference on Hexagon only available in ARM64 environment
        # Change processor to CPU if environment is incorrect
        if "arm64" not in self.arch and "aarch64" not in self.arch:
            self.processor = "CPU" #if "ARM64" not in self.arch else self.processor

        self


if __name__=="__main__":
    
    model_name = "hrnet_pose"
    system = "windows"
    processor = "npu"
    model_type = "quantized"
    iLoad = ModelLoader(model=model_name, processor=processor, model_type=model_type)
    onnx_graph = "hrnet_quantized.onnx"#
    print(iLoad.graphs)
    session = iLoad.load_model(onnx_graph)
    print(session.get_providers())
   
    print(session.get_inputs()[0])
    # print(iLoad.executioners)
    print(session.get_providers())
    
