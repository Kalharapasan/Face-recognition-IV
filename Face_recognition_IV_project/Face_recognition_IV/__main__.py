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
    def __init__(self):
        self.data_dir = "data"
        self.model_path = "face_recognizer_model.xml"
        self.users_config = "users.json"
        self.screenshots_dir = "screenshots"
        
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        self.recognizer = None

        self.create_directories()
        
        self.users = self.load_users_config()
        
        print("🚀 Face Recognition System initialized!")

    def create_directories(self):
        directories = [self.data_dir, self.screenshots_dir]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Created directory: {directory}")
    
    def load_users_config(self):
        if os.path.exists(self.users_config):
            try:
                with open(self.users_config, 'r') as f:
                    users = json.load(f)
                print(f"📋 Loaded {len(users)} users from configuration")
                return users
            except Exception as e:
                print(f"⚠️ Error loading users config: {e}")
        

        default_users = {"1": "User1"}
        self.save_users_config(default_users)
        print("📋 Created default users configuration")
        return default_users
    
    def save_users_config(self, users=None):
        if users is None:
            users = self.users
        
        try:
            with open(self.users_config, 'w') as f:
                json.dump(users, f, indent=2)
            print("💾 Users configuration saved")
        except Exception as e:
            print(f"❌ Error saving users config: {e}")
    
    