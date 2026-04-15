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
    
# ============================================================================
# STEP 3: USER MANAGEMENT
# ============================================================================

    def add_user(self, user_id, name):
        try:
            user_id = str(user_id)
            self.users[user_id] = name
            self.save_users_config()
            print(f"✅ Added user: {name} (ID: {user_id})")
            return True
        except Exception as e:
            print(f"❌ Error adding user: {e}")
            return False
    
    def list_users(self):
        print("\n👥 REGISTERED USERS:")
        print("-" * 30)
        if len(self.users) == 0:
            print("No users registered")
        else:
            for user_id, name in self.users.items():
                print(f"  ID: {user_id} - Name: {name}")
        print()
    
    def remove_user(self, user_id):
        user_id = str(user_id)
        if user_id in self.users:
            name = self.users[user_id]
            del self.users[user_id]
            self.save_users_config()
            print(f"✅ Removed user: {name} (ID: {user_id})")
            return True
        else:
            print(f"❌ User ID {user_id} not found")
            return False

# ============================================================================
# STEP 4: FACE DETECTION
# ============================================================================

    def detect_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Return the largest face if multiple detected
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        
        x, y, w, h = faces[0]
        cropped_face = img[y:y+h, x:x+w]
        return cropped_face
    
    def test_camera(self):
        print("📹 Testing camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot access camera")
            return False
        
        print("✅ Camera accessible")
        print("📸 Press any key to take test photo, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Cannot read frame")
                break
            
            # Add text overlay
            cv2.putText(frame, "Camera Test - Press any key", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Camera Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key != 255:  # Any other key pressed
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"camera_test_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Test photo saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        return True

# ============================================================================
# STEP 5: DATA COLLECTION FOR TRAINING
# ============================================================================
    def collect_training_data(self, user_id, num_samples=200):