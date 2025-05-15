from setuptools import setup, find_packages

setup(
    name="/Users/colinlinke/Documents/AI in Industry Project/AI_Industry_Project.nosync",
    packages=find_packages(),
)

import torch

# Basic check
print(f"CUDA available: {torch.cuda.is_available()}")

# More detailed information
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")  # Returns device index
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"CUDA version: {torch.version.cuda}")