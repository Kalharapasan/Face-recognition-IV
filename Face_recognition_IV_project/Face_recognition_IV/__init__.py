#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime

def check_libraries():
    """Check if all required libraries are installed"""
    print("🔍 Checking required libraries...")
    
    try:
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found. Install with: pip install opencv-python")
        return False
    
    try:
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy not found. Install with: pip install numpy")
        return False
    
    try:
        from PIL import Image
        print("✅ PIL/Pillow: Available")
    except ImportError:
        print("❌ PIL not found. Install with: pip install Pillow")
        return False
    
    print("✅ All libraries are available!")
    return True

class FaceRecognitionSystem: