# ONNX Runtime Sample Apps for Qualcomm Hexagon NPU
This repository contains sample apps for running ONNX models efficiently using [ONNX Runtime](https://onnxruntime.ai/), specifically targeting Qualcomm Hexagon NPU with [QNN Execution Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
## Table of Contents
1. [Available Apps](#available-apps)
2. [Getting Started](#getting-started)
3. [Quick Start](#quick-start)
4. [Contributing](#contributing)
5. [Testing](#testing)
6. [Directory Structure](#directory-structure)
7. [License](#license)

## Available Apps
| App Name | Model Used | Providers | Quick Start | Notebook | Blog | Video |
|----------|------------|-----------|-------------|----------|-------|------|
| HRNet Pose Detection | HRNetPose | QNN/CPU | [Complete](#quick-start) | [Complete](./notebooks/pose_detection/) | [Blog](https://www.qualcomm.com/developer/blog/2025/03/enable-pose-detection-snapdragon-x-elite-step-by-step-tutorial) | [Youtube](https://youtu.be/OASSOhlSpfY?si=gNJLRHAxpl4IUflv) |
| DeepSeek Local | DeepSeek | QNN/CPU | [Complete](#quick-start) | [Complete](./notebooks/llm/) | [Blog](https://www.qualcomm.com/developer/project/5-part-series--enable-deepseek-on-snapdragon-x-elite)  | [Youtube](https://www.youtube.com/watch?v=VRDB_ob7ulA) |
| Gemma-3 Local | Gemma-3_1B | CPU | [Complete](#quick-start) | [Complete](./notebooks/llm) | ❌ | ❌

These apps demonstrate end-to-end inference using ONNX Runtime on devices with Hexagon NPUs. Each app includes:
- Input preprocessing
- Onnx model inference
- Output postprocessing
  
Supported features:
- CPU fallback (if you don't have access to Hexagon NPU)
- Hexagon QNN Execution

## Getting Started
#### General Requirements
- Python (version 3.11.+)
   - If targeting Hexagon ensure you install ARM64 compatible Python version
- C and C++ support in Visual Studio
   - [Installation Instructions](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)
- Rust Support
   - [Installation Instructions](https://rustup.rs/)
 - Git Support
   - [Installation Instructions](https://git-scm.com/downloads/win)
#### 1. Clone the Repository
```
git clone https://github.com/DerrickJ1612/qnn_sample_apps.git
```
#### 2. Setup Virtual Environment
```
python -m venv env_qnn_sample_apps
```
```
env_qnn_sample_apps\Scripts\Activate.ps1
```
DeepSeek Example:
```
pip install -r \src\deepseek_r1\requirements.txt
```
#### 3. Download Models
| Model Name  | Description              | Download Source                                                                                                                                                                                                            |
|-------------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HRNetPose   | Human pose estimation    | [AI Hub](https://aihub.qualcomm.com/compute/models/hrnet_pose?domain=Computer+Vision&useCase=Pose+Estimation)                                                                                                              |
| DeepSeek R1 | Reasoning Language Model | [Microsoft AI Toolkit](https://marketplace.visualstudio.com/items?itemName=ms-windows-ai-studio.windows-ai-studio); [Google Drive](https://drive.google.com/drive/folders/1hCopYw7rMdeOm3zV6NC2do9orzpKqAMf?usp=sharing)   | 
| Gemma-3 1B | Instruction Model | [Hugging Face](https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX-GQA); [Google Drive](https://drive.google.com/drive/folders/1hCopYw7rMdeOm3zV6NC2do9orzpKqAMf?usp=sharing) |

- After downloading model move them to: 

  - `qnn_sample_apps/models/<subdirectory_of_model>`

**Note:** You only need to move files ending in *.onnx and *.bin/*.data


## Quick Start

| App Name               | CLI Command                                  |
|------------------------|----------------------------------------------|
| 'HRNet Pose Detection' | `python ./src/hrnet_pose/main.py `       |
| 'Local LLM'       | `python ./src/llm/main.py --help` |

## Contributing
I welcome contributions to this repository! Please refer to our [contributing guide](CONTRIBUTING.md) for how to contribute.

## Testing
- All regression tests must pass
- New features should include appropriate test coverage
```
cd ./qnn_sample_apps
pytest -vv
```
## License
This project is licensed under the [MIT](https://github.com/DerrickJ1612/qnn_sample_apps/blob/main/LICENSE.txt)

## Directory Structure
```
QNN_SAMPLE_APPS/
├── models/
│   ├── hrnet_pose/
│   ├── cpu-deepseek-r1-distill-qwen-7b/
│   ├── qnn-deepseek-r1-distill-qwen-1.5b/
│   ├── qnn-deepseek-r1-distill-qwen-7b/
│   ├── qnn-deepseek-r1-distill-qwen-14b/
│   ├── gemma-3-1b-it-ONNX-GQA/
├── notebooks/
│   ├── llm/
│   ├── pose_detection/
├── scripts/
├── src/
│   ├── llm/
│   ├── hrnet_pose/
│   └── model_loader.py
├── tests/
├── CODE_OF_CONDUCT.md
├── conftest.py
├── CONTRIBUTING.md
├── desktop.ini
├── executioner.json
├── LICENSE.txt
├── models.json
├── pyproject.toml
├── README.md
└── setup.py
```


