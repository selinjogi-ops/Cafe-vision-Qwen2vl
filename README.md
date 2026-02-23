# Cafe Vision Qwen2VL

Real-time Cafe Monitoring System using YOLOv8, DeepSort Tracking, and Qwen2-VL-2B-Instruct for multimodal scene understanding.

# Overview

Cafe Vision is a real-time AI surveillance and scene analysis system designed for smart cafe environments.

The system combines:

*Object Detection (People)

*Multi-object Tracking

*Vision-Language Reasoning

*Asynchronous AI Inference

*CSV Logging of Observations

It detects people using YOLO, tracks them with DeepSort, and periodically asks a Vision-Language Model to describe what is happening inside the cafe.

# System Architecture

<img width="373" height="242" alt="image" src="https://github.com/user-attachments/assets/d4d397d1-66b2-4175-bb73-15af8b08ba42" />

# Models Used

# 1)YOLOv8

*Model: yolov8n.pt

*Purpose: Real-time person detection

*Framework: Ultralytics

# 2)DeepSort

*Real-time multi-object tracking

*Maintains consistent person IDs

*Handles occlusion & re-identification

# 3)Alibaba Cloud – Qwen2-VL-2B-Instruct

*Model ID: Qwen/Qwen2-VL-2B-Instruct

*Type: Vision-Language Model

*Framework: HuggingFace Transformers

*Generates contextual descriptions of cafe activity

# Key Features

*Real-time person detection

*Unique ID tracking per individual

*Asynchronous VLM processing (non-blocking UI)

*Scene understanding every ~5 seconds

*CSV log file with timestamps

*GPU support (automatic CUDA detection)

*CPU fallback supported

# Async VLM Design

The Vision-Language Model runs in a separate thread to prevent frame freezing.

*threading.Thread() used for async inference

*Global state prevents overlapping model calls

*Frame copied before sending to model

*Results displayed in overlay text

This ensures smooth real-time performance.

# Installation

Install PyTorch (CPU Example):

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

For GPU users (CUDA):

pip install torch torchvision

Install Dependencies:

pip install -r requirements.txt

Run the System

python cafe8(Qwen2 VL).py

Press ESC to exit.

# Output

*Bounding boxes around detected persons

*Unique ID tracking (ID 1, ID 2, etc.)

*Live Qwen2-VL description overlay

*Automatic CSV log file:

# Hardware Requirements

Minimum:

*8GB RAM

*Webcam (720p or higher)

Recommended:

*NVIDIA GPU (for faster Qwen2-VL inference)

*16GB RAM

# Research Significance

This project demonstrates:

*Hybrid architecture (Detection + Tracking + VLM)

*Real-time multimodal reasoning

*Edge-device compatible VLM deployment

*Async AI inference in live CV systems

*Practical smart surveillance prototype

# Future Improvements

*Replace yolov8n with yolov8m for higher accuracy

*Add crowd density estimation

*Integrate emotion analysis

*Store logs in database instead of CSV

*Deploy as Flask / FastAPI dashboard

*Add anomaly detection logic
