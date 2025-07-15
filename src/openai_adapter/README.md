# ONNX Runtime Sample Apps for Qualcomm Hexagon NPU
This directory contains an OpenAI API adapter which provides a compatibility layer for LLM inference. This will allow you to use OpenAI's API format with your custom inference backend

## Overview
The OpenAI adapter serves as a bridge between OpenAI's API specification and the LLMs provided within this repo.
- ### API Compatibility
    - Full Support for OpenAI's chat completions API Format
- ### Easy Integration
    - Simple instantiation and server startup
- ### Flexible LLM Utilization
    - Choose from any LLM currently provided within this repo 🚧

## Directory Structure

openai_adapter/
├── openai_api_adapter.py                   # Main adapter implementation and server initialization
├── example_openai_chat_completion.py       # After starting server above this provides an OpenAI integration example
└── README.md                       # This file