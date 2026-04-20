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
        print(f"📸 Starting data collection for User ID: {user_id}")
        print(f"🎯 Target samples: {num_samples}")
        print("\n💡 Instructions:")
        print("  - Look directly at the camera")
        print("  - Keep your face clearly visible")
        print("  - Move your head slightly for different angles")
        print("  - Ensure good lighting")
        print("  - Press 'q' to quit early")
        if str(user_id) not in self.users:
            print(f"❌ User ID {user_id} not found. Please add user first.")
            return False
        
        user_name = self.users[str(user_id)]
        print(f"👤 Collecting data for: {user_name}")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot access camera")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        img_count = 0
        print("\n🎬 Data collection started!")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Cannot read frame")
                break

            # Create display frame
            display_frame = frame.copy()
            
            # Add information overlay
            cv2.putText(display_frame, f"User: {user_name} (ID: {user_id})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Samples: {img_count}/{num_samples}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'q' to quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show original frame
            cv2.imshow("📸 Data Collection - Camera Feed", display_frame)

            # Detect and process face
            cropped_face = self.detect_face(frame)
            
            if cropped_face is not None:
                img_count += 1
                
                # Resize to standard size
                face_resized = cv2.resize(cropped_face, (200, 200))
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                
                # Save image
                filename = f"{self.data_dir}/user.{user_id}.{img_count}.jpg"
                cv2.imwrite(filename, face_gray)
                
                # Show processed face
                display_face = face_gray.copy()
                cv2.putText(display_face, f"Sample: {img_count}/{num_samples}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("✅ Processed Face", display_face)

                # Progress update
                if img_count % 20 == 0:
                    print(f"📊 Progress: {img_count}/{num_samples} samples collected")
                
                # Stop when target reached
                if img_count >= num_samples:
                    print(f"🎉 Target reached! Collected {img_count} samples")
                    break
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"⏹️ Collection stopped by user at {img_count} samples")
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        if img_count > 0:
            print(f"✅ Data collection completed!")
            print(f"📊 Collected {img_count} samples for {user_name}")
            print(f"📁 Files saved in '{self.data_dir}' directory")
            return True
        else:
            print("❌ No samples collected")
            return False
        
# ============================================================================
# STEP 6: MODEL TRAINING
# ============================================================================

    def train_model(self):
        print("🧠 Starting model training...")
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            print(f"❌ Data directory '{self.data_dir}' not found")
            print("💡 Collect training data first")
            return False
        
        # Get all training images
        image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        
        if len(image_files) == 0:
            print("❌ No training images found")
            print("💡 Collect training data first")
            return False
        
        print(f"📊 Found {len(image_files)} training images")

        # Prepare training data
        faces = []
        labels = []
        
        print("📝 Processing training images...")
        
        successful_images = 0
        failed_images = 0

        for image_file in image_files:
            try:
                # Load image
                image_path = os.path.join(self.data_dir, image_file)
                img = Image.open(image_path).convert('L')  # Convert to grayscale
                
                # Convert to numpy array
                face_np = np.array(img, 'uint8')
                
                # Extract user ID from filename (user.ID.number.jpg)
                parts = image_file.split('.')
                if len(parts) >= 3:
                    user_id = int(parts[1])
                else:
                    print(f"⚠️ Skipping {image_file}: Invalid filename format")
                    failed_images += 1
                    continue
                # Add to training data
                faces.append(face_np)
                labels.append(user_id)
                successful_images += 1
                
                # Show progress
                if successful_images % 50 == 0:
                    print(f"   📊 Processed {successful_images}/{len(image_files)} images")
                
            except Exception as e:
                print(f"⚠️ Error processing {image_file}: {e}")
                failed_images += 1
                continue
            
        if successful_images == 0:
            print("❌ No valid training images found")
            return False
        
        print(f"✅ Successfully processed {successful_images} images")
        if failed_images > 0:
            print(f"⚠️ Failed to process {failed_images} images")
        
        # Convert to numpy arrays
        faces = np.array(faces)
        labels = np.array(labels)

        # Show training summary
        unique_users = len(set(labels))
        print(f"👥 Training data for {unique_users} different users")

        # Count samples per user
        user_counts = {}
        for label in labels:
            user_counts[label] = user_counts.get(label, 0) + 1
        
        print("📊 Samples per user:")
        for user_id in sorted(user_counts.keys()):
            count = user_counts[user_id]
            name = self.users.get(str(user_id), f"User {user_id}")
            print(f"   👤 {name} (ID: {user_id}): {count} samples")
        
        # Create and train the recognizer
        print("🔄 Training LBPH Face Recognizer...")
        
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(faces, labels)

        # Save the trained model
        self.recognizer.write(self.model_path)
        
        print(f"✅ Model training completed successfully!")
        print(f"💾 Model saved as: {self.model_path}")
        
        # Show model file info
        model_size = os.path.getsize(self.model_path) / 1024  # KB
        print(f"📊 Model file size: {model_size:.1f} KB")
        
        return True

# ============================================================================
# STEP 7: FACE RECOGNITION
# ============================================================================

    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model file '{self.model_path}' not found")
            print("💡 Train the model first")
            return False
        
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(self.model_path)
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def start_recognition(self, confidence_threshold=75):
        """
        Start real-time face recognition
        """
        print(f"🔍 Starting face recognition...")
        print(f"🎯 Confidence threshold: {confidence_threshold}%")
        print("📹 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save screenshot")
        print("   - Press 'c' to change confidence threshold")

        if self.recognizer is None:
            if not self.load_model():
                return False
        
        